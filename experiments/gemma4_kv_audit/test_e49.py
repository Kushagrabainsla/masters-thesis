"""Tests for E-49 (no GPU): the layout identity, the fit, every decision path on stub data, and a stub session.

Run: uv run --no-project --with numpy --with pytest python -m pytest test_e49.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import analyze_e49 as an

HERE = Path(__file__).resolve().parent
TEL = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"
CELLS = {"m31": ("RedHatAI/gemma-4-31B-it-FP8-dynamic", "2048:4,8192:4,32768:4,65536:4", "2048:4,32768:4,1024:1,1024:8,1024:32"),
         "m26": ("google/gemma-4-26B-A4B-it", "2048:8,8192:8,32768:8,65536:8", "2048:8,32768:8,1024:1,1024:8,1024:32")}


def stub(res: Path, key: str, *extra: str) -> None:
    model, cells, prof = CELLS[key]
    subprocess.run([sys.executable, str(HERE / "decode_probe.py"), "--stub", "--model", model, "--out", str(res / key),
                    "--cells", cells, "--waste-cells", "1024:1,1024:8,1024:32", "--profile-cells", prof, *extra],
                   check=True, capture_output=True)


def env(res: Path, copy: float = 3000.0, throttled: bool = False) -> None:
    e = res / "env-1"
    e.mkdir(parents=True, exist_ok=True)
    reason = "0x8" if throttled else "0x0"
    (e / "telemetry.csv").write_text(TEL + "".join(f"2026/10/30 10:00:00.000, 600, 1980, 60, 99, {reason}\n" for _ in range(150)))
    (e / "microbench_start.json").write_text(json.dumps({"copy_gb_per_s_median": copy}))


def decision(res: Path) -> str:
    return an.decide(an.summarize(res))


def test_layout_identity() -> None:
    sys.path.insert(0, str(HERE))
    import kv_layout_check as kv

    r = kv.check(seed=1, keys=500)
    assert r["values_per_key_compact"] == 640 and abs(r["saving"] - 0.375) < 1e-12
    assert r["max_rel_err_float64"] < 1e-12
    assert r["max_rel_err_bf16_compact"] < 3 * r["max_rel_err_bf16_standard"] + 1e-6


def test_fit_exact_line() -> None:
    f = an.fit([(x, 10 + 0.5 * x) for x in (1, 2, 3, 4, 5)])
    assert abs(f["b"] - 0.5) < 1e-12 and abs(f["c"] - 10) < 1e-9 and f["r2"] > 0.999999


def test_go(tmp_path: Path) -> None:
    for k in CELLS:
        stub(tmp_path, k)
    env(tmp_path)
    s = an.summarize(tmp_path)
    assert all(m["valid"] for m in s["models"].values()), s["models"]["m31"].get("why")
    assert an.decide(s).startswith("GO (")
    m31 = s["models"]["m31"]["decide"]
    assert 0.08 < m31["32768"]["S"] < 0.11 and 0.12 < m31["65536"]["S"] < 0.16   # close to the byte model


def test_kill_when_attention_is_tiny(tmp_path: Path) -> None:
    for k in CELLS:
        stub(tmp_path, k, "--stub-bw-gbps", "1e6")
    env(tmp_path)
    assert decision(tmp_path).startswith("KILL")


def test_pivot_kernel_bound(tmp_path: Path) -> None:
    for k in CELLS:
        stub(tmp_path, k, "--stub-bw-gbps", "800")
    env(tmp_path)
    d = decision(tmp_path)
    assert d.startswith("PIVOT") and "KERNEL-BOUND" in d


def test_pivot_attribution(tmp_path: Path) -> None:
    for k in CELLS:
        stub(tmp_path, k, "--stub-bad-attribution")
    env(tmp_path)
    s = an.summarize(tmp_path)
    assert s["models"]["m31"]["V4b"] is False and s["models"]["m31"]["valid"]
    assert an.decide(s).startswith("PIVOT")


def test_invalid_paths(tmp_path: Path) -> None:
    for k in CELLS:
        stub(tmp_path / "acc", k, "--stub-accuracy", "0.2")
        stub(tmp_path / "adv", k, "--stub-bad-advance")
        stub(tmp_path / "tel", k)
    env(tmp_path / "acc")
    env(tmp_path / "adv")
    env(tmp_path / "tel", throttled=True)
    assert decision(tmp_path / "acc").startswith("INVALID") and "V6" in decision(tmp_path / "acc")
    assert "V2" in decision(tmp_path / "adv")
    assert decision(tmp_path / "tel") == "INVALID (V8 telemetry)"


def test_one_model_and_noise(tmp_path: Path) -> None:
    stub(tmp_path / "one", "m31")
    env(tmp_path / "one")
    assert decision(tmp_path / "one").startswith("GO (one model)")
    stub(tmp_path / "noisy", "m31", "--stub-noise", "0.05")
    stub(tmp_path / "noisy", "m26", "--stub-noise", "0.05")
    env(tmp_path / "noisy")
    s = an.summarize(tmp_path / "noisy")
    assert not s["models"]["m31"]["checks"]["V7"] or not s["models"]["m31"]["checks"]["V3"]


def test_stub_session(tmp_path: Path) -> None:
    res = tmp_path / "e49"
    e = dict(os.environ, STUB="1", RES=str(res), CODE=str(HERE.parent), AUTO_STOP="0")
    sh = shutil.which("sh")
    r = subprocess.run([sh, str(HERE / "session_e49.sh")], env=e, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    log = (res / "session.log").read_text()
    assert "m31: exit 0" in log and "m26: exit 0" in log and "decision: GO (" in log
    r = subprocess.run([sh, str(HERE / "session_e49.sh")], env=e, capture_output=True, text=True, timeout=300)
    assert "m31: ok before, skip" in (res / "session.log").read_text()

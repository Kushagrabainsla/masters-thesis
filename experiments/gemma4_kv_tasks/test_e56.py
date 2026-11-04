"""E-56 tests (README Section 12). Run: uv run --no-project --with numpy --with torch --with pytest --with tokenizers
python -B -m pytest -q -p no:cacheprovider experiments/gemma4_kv_tasks/test_e56.py"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_model"), str(EXP / "ledger")]
import analyze_e56 as an  # noqa: E402
import run_e56 as run  # noqa: E402
import splice_e55 as sp  # noqa: E402

DATA = HERE / "data_e56.json"


def test_data_file_is_the_frozen_plan():
    data = json.loads(DATA.read_text())
    assert an.sha_file(DATA) == an.DATA_SHA256
    assert data["model_id"] == "google/gemma-4-31B-it" and data["revision"] == an.REVISION
    assert len(data["gsm8k"]) == 400 and len({g["row"] for g in data["gsm8k"]}) == 400
    assert len(data["needle"]) == 32 and len({n["gold"] for n in data["needle"]}) == 32
    assert [h["source"] for h in data["health"]] == ["B1", "P1", "C1"]
    assert all(len(h["ids"]) == 2_048 for h in data["health"])
    lim = run.ENGINE["max_model_len"]
    for g in data["gsm8k"]:
        assert g["score_ids"][:len(g["ids"])] == g["ids"] and g["answer_start"] == len(g["ids"])
        assert g["score_ids"][-1] == data["eot_id"] and an._num(g["gold"]) is not None
        assert len(g["ids"]) + run.GSM_MAX_TOKENS <= lim and len(g["score_ids"]) <= lim
    assert all(len(n["ids"]) + run.NEEDLE_MAX_TOKENS <= lim for n in data["needle"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HERE / "data/tokenizer/tokenizer.json"))
    for n in data["needle"][::7]:
        text = tok.decode(n["ids"], skip_special_tokens=False)
        assert f"The secret passphrase for the archive is {n['gold']}." in text and text.endswith("<channel|>")


def test_answer_extraction_and_grading():
    assert an.extract_gsm("x = 3\n#### 1,234") == ("1234", "####")
    assert an.extract_gsm("so #### $5.50 total") == ("5.50", "####")
    assert an.extract_gsm("The answer is 42.") == ("42", "last")
    assert an.extract_gsm("no number") == (None, "none")
    assert an.gsm_correct("16.0", "16") and not an.gsm_correct("17", "16") and not an.gsm_correct(None, "16")
    assert an.extract_needle(" The passphrase is Cobalt-4980.") == "cobalt-4980"
    assert an.extract_needle("I do not know.") == "i do not know"
    assert an.same_answer("18", "18.00") and not an.same_answer("18", "19") and an.same_answer("x-1234", "x-1234")


def test_paired_interval_covers_the_true_difference():
    """The Agresti-Min interval at the E-56 size (432 items, 88% right, about 5% discordant) covers 93% to 98%."""
    rng = np.random.default_rng(56)
    for true_d in (0.0, -0.01, -0.03):
        p10, p01 = (0.05 - true_d) / 2, (0.05 + true_d) / 2          # x right y wrong; x wrong y right
        probs = [0.88 - p10, p10, p01, 1 - 0.88 - p01]               # cells 11, 10, 01, 00
        hits = 0
        for _ in range(2_000):
            cell = rng.choice(4, size=432, p=probs)
            x, y = np.isin(cell, (0, 1)), np.isin(cell, (0, 2))
            _, (lo, hi) = an.paired_diff(list(x), list(y))
            hits += lo <= true_d <= hi
        assert 0.93 <= hits / 2_000 <= 0.99, (true_d, hits)


def test_ratio_interval_covers_delta():
    rng = np.random.default_rng(57)
    hits = 0
    for _ in range(2_000):
        n = rng.integers(40, 160, 400).astype(float)
        s = 0.001 * n + rng.normal(0, 0.05, 400) * np.sqrt(n)
        _, (lo, hi) = an.ratio_interval(s, n)
        hits += lo <= 0.001 <= hi
    assert 0.93 <= hits / 2_000 <= 0.97


def test_decision_rule_gives_each_outcome():
    def st(acc, dis, nll):
        return {"acc_ci": acc, "dis_ci": dis, "nll_ci": nll}
    assert an.decide(st((-0.02, 0.02), (-0.01, 0.02), (-0.001, 0.001))) == ("GO1", "GO2", "GO3")
    assert an.decide(st((-0.05, -0.01), (0.04, 0.09), (0.012, 0.02))) == ("KILL1", "WORSE2", "KILL3")
    assert an.decide(st((-0.04, 0.01), (0.0, 0.05), (-0.001, 0.004))) == ("INCONCLUSIVE1", "INCONCLUSIVE2",
                                                                         "INCONCLUSIVE3")
    assert an.decide(st((-0.01, 0.035), None, (-0.003, 0.0)))[:2] == ("INCONCLUSIVE1", "NA2")
    assert an.decide(st((-0.02, 0.02), (-0.01, 0.02), (0.0047, 0.0053)))[2] == "SMALL3"      # (A1, R-79 F7)


def test_health_gate_stops_a_broken_baseline():
    data = json.loads(DATA.read_text())
    good = {x["id"]: {"correct": True} for x in data["gsm8k"] + data["needle"]}
    ref = {g["id"]: (-0.4 * 100, 100) for g in data["gsm8k"]}
    assert an.health_problems(good, ref, data)[0] == []
    bad = dict(good, **{x["id"]: {"correct": i % 2 == 0} for i, x in enumerate(data["gsm8k"])})
    assert any("GSM8K accuracy" in p for p in an.health_problems(bad, ref, data)[0])
    ref_bad = {g["id"]: (-9.0 * 100, 100) for g in data["gsm8k"]}       # an E-55-like baseline: 9 nats
    assert any("reference NLL" in p for p in an.health_problems(good, ref_bad, data)[0])
    ref_style = {g["id"]: (-2.0 * 100, 100) for g in data["gsm8k"]}     # (A1, R-79 B1) a healthy model, other style
    assert an.health_problems(good, ref_style, data)[0] == []


def test_preflight_stops_on_the_decision_arms_and_records_stock8():
    """(A1, R-79 F2) a STOCK8 problem in the pre-flight is recorded and does not stop the run."""
    lps = {a: np.full(8, -1.0 - 0.01 * i, np.float32) for i, a in enumerate(an.ARMS)}
    lps["DET"] = lps["BF16"].copy()
    ev = {"checks": {a: [] for a in an.ARMS}, "hook_errors": {},
          "counts": {f"{a}|PRE|{lay}": 8 for a in an.ARMS for lay in range(10)},
          "gaps": {a: {lay: 0.002 for lay in range(10)} for a in an.ARMS},
          "gaps_id": {a: {lay: 0.05 for lay in range(10)} for a in an.ARMS},
          "seps": {a: {lay: (0.0 if a == "BF16" else 0.05) for lay in range(10)} for a in an.ARMS}}
    ev["gaps"]["STOCK8"][3] = 0.5
    assert an.preflight_problems(lps, ev, 10, 8) == []
    assert any("STOCK8" in p for p in an.preflight_problems(lps, ev, 10, 8, arms=("STOCK8",)))


def test_session_limits_fit_the_cost_cap():
    """The session watchdog fires before the boot watchdog; the worst case (15 min to SSH, then the boot watchdog)
    stays inside the cap of $2.50 at $1.79/h (the A100 80GB live price, 2026-11-04). (A1) The SSH limit is 600 s."""
    text = (HERE / "session_e56.sh").read_text()
    limit = int(re.search(r"POD_LIMIT:-(\d+)", text).group(1))
    boot = int(re.search(r"BOOT_CAP:-(\d+)", text).group(1))
    assert limit + 150 <= boot
    assert (600 + boot) / 3600 * 1.79 <= 2.50
    assert int(re.search(r"AUTO_STOP_DELAY:-(\d+)", text).group(1)) >= 300


def run_stub(tmp: Path, args: str = "", extra_env: dict | None = None, check: bool = True) -> dict:
    res = tmp / "res"
    env = dict(os.environ, STUB="1", RES=str(res), CODE=str(EXP), STUB_ARGS=args, AUTO_STOP="0", POD_LIMIT="900",
               PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    env.update(extra_env or {})
    subprocess.run(["sh", str(HERE / "session_e56.sh")], env=env, check=check, capture_output=True, timeout=900)
    return json.loads((res / "summary.json").read_text())


def test_stub_session_runs_end_to_end(tmp_path):
    """The stub engine carries the real hooks in the engine's call shapes (L-26): the probe gives every arm its gaps
    in every layer, the batched calls reach every layer with at least the expected tokens, DET reproduces BF16, and
    the frozen rule decides."""
    s = run_stub(tmp_path)
    assert s["decision"] == "GO1 GO2 GO3", s.get("validity")
    assert not any(s["validity"].values()) and s["excluded"] == {} and s["preflight"] == []
    assert s["det"]["ok"] and s["det"]["bit_identical"]
    rec = json.loads((tmp_path / "res/run_main.json").read_text())
    assert {a: len(g) for a, g in rec["gaps"].items()} == {a: 10 for a in an.ARMS}
    assert sum(x > 0.001 for x in rec["seps"]["BF16C"].values()) == 9 and rec["seps"]["BF16C"]["29"] == 0.0
    assert rec["health"]["problems"] == [] and set(s["stats"]["accuracy"]) == set(an.ARMS) | {"DET"}
    assert set(s["stats"]["nll_long"]) == {"BF16C", "BF16R", "STOCK8"} and s["stats"]["gen"]["BF16"]["hash_marks"] == 1.0
    assert s["det"]["answer_ids_differ"] == 0.0 and rec["preflight"]["stock8_problems"] == []
    sha = (tmp_path / "res/code_sha.txt").read_text()
    assert all(sha.count(f) == 1 for f in ("splice_e55.py", "run_e55.py", "analyze_e55.py", "run_e56.py"))


def test_stub_session_finds_planted_failures(tmp_path):
    s = run_stub(tmp_path / "a", "--stub-loss BF16C=0.08 --stub-plant step_error")
    assert s["H1"] == "KILL1", s["stats"]["acc_ci"]
    assert [e["step"] for e in s["step_errors"]] == ["STOCK8"] and "STOCK8" not in s["stats"]["accuracy"]
    s = run_stub(tmp_path / "b", "--stub-plant broken_baseline")                  # V8 stops it after BF16 (L-29)
    rec = json.loads((tmp_path / "b/res/run_main.json").read_text())
    assert s["decision"] == "INVALID" and "V8" in s["failed"] and list(rec["arms"]) == ["BF16"]
    assert "HealthError" in rec["errors"][0]
    s = run_stub(tmp_path / "c", "--stub-plant ignore_hook")                      # V7 in the pre-flight
    rec = json.loads((tmp_path / "c/res/run_main.json").read_text())
    assert s["decision"] == "INVALID" and rec["arms"] == {} and "PreflightError" in rec["errors"][0]
    assert any("nearer the unchanged reference" in p for p in rec["preflight"]["problems"])
    s = run_stub(tmp_path / "d", "--stub-plant det_noise")                        # DET in the pre-flight
    rec = json.loads((tmp_path / "d/res/run_main.json").read_text())
    assert s["decision"] == "INVALID" and any("DET differs" in p for p in rec["preflight"]["problems"])
    s = run_stub(tmp_path / "e", "--stub-plant det_ids")                          # (A1, R-79 F1) recorded only
    assert s["decision"] == "GO1 GO2 GO3" and s["det"]["same_ids"] == s["det"]["answers"] - 1


def test_time_cap_cuts_stock8_and_keeps_the_decision(tmp_path):
    s = run_stub(tmp_path, "", {"POD_LIMIT": "70", "AN_RESERVE": "10"})        # (A1, R-79 F5) a certain cut
    assert [c["step"] for c in s["cut"]] == ["STOCK8"] and s["decision"] == "GO1 GO2 GO3"


def test_sigterm_after_the_decision_arms_keeps_the_decision(tmp_path):
    res = tmp_path / "res"
    p = subprocess.Popen([sys.executable, "-B", str(HERE / "run_e56.py"), "--out", str(res), "--stub",
                          "--stub-seconds", "0.01", "--deadline", str(time.time() + 3_600)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            if json.loads((res / "run_main.json").read_text()).get("decision_arms_done"):
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=120) == 1
    rec = json.loads((res / "run_main.json").read_text())
    assert rec["status"] == "error" and "Terminated" in rec["errors"][0] and rec["decision_arms_done"]
    assert an.summarize(res, DATA)["decision"] == "GO1 GO2 GO3"


def test_v2_reads_the_engine_record_of_this_vllm_build(tmp_path):
    """L-28: the eager check uses the values the real engine recorded in E-55 ("0" and "NONE"); the code hashes are
    compared with the design."""
    data = json.loads(DATA.read_text())
    main = {"stub": False, "gpu": "NVIDIA A100-SXM4-80GB", "status": "ok", "tok_check": {"ok": True},
            "hooks": {"hooked": []}, "engine": {"kv_cache_dtype": "auto", "prefix_caching": False,
                                                "compilation_mode": "0", "cudagraph_mode": "NONE",
                                                "max_num_batched_tokens": an.ENGINE_TOKENS}}
    (tmp_path / "code_manifest.json").write_text(json.dumps({n: "a" * 64 for n in an.CODE_FILES}))
    (tmp_path / "code_sha.txt").write_text("".join(f"{'a' * 64}  {n}\n" for n in an.CODE_FILES))
    v, _ = an.validity(tmp_path, data, main, {}, {}, {}, rehearsal=True)
    assert v["V2"] == [] and v["V1"] == []
    main["engine"]["compilation_mode"] = "3"
    (tmp_path / "code_sha.txt").write_text("".join(f"{'b' * 64}  {n}\n" for n in an.CODE_FILES))
    v, _ = an.validity(tmp_path, data, main, {}, {}, {}, rehearsal=True)
    assert "the engine is not eager" in v["V2"] and sum("code hash" in p for p in v["V2"]) == len(an.CODE_FILES)


def test_make_manifest_hashes_the_committed_files():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_manifest_e56", HERE / "make_manifest.py")
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)          # E-55 has a file of the same name
    try:
        m = mm.manifest("HEAD")
    except subprocess.CalledProcessError:
        pytest.skip("the E-56 files are not committed yet")
    assert set(m) == set(an.CODE_FILES) and m["data_e56.json"] == an.DATA_SHA256


def test_manifest_is_accepted_by_the_ledger(tmp_path):
    import ledger
    s = run_stub(tmp_path)
    m = an.manifest(s, tmp_path / "res", "0" * 40, "e56-test")
    m["hardware"].update(driver="stub", cuda="stub")
    assert ledger.validate(m, set()) == []

"""Tests for E-49 (no GPU): the layout identity, the fit, every decision path on stub data, the kernel-name
classes, the probe loop on a fake engine, and stub sessions (happy path, failed run, time cap, rerun) from a
clean copy of the upload folders.

Run: uv run --no-project --with numpy --with pytest python -m pytest test_e49.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import analyze_e49 as an

HERE = Path(__file__).resolve().parent
UPLOAD = ["gemma4_kv_audit", "safedispatch_testA2", "runpod", "fp8_quant_fusion", "nvfp4_dispatch", "safedispatch_testA",
          "ledger"]   # README Section 11: the folders the Pod needs
TEL = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"
CELLS = {"m31": ("RedHatAI/gemma-4-31B-it-FP8-dynamic", "2048:4,8192:4,32768:4,65536:4,65536:4:1,32768:4:1",
                 "1024:1,1024:8,1024:32,16384:8", "2048:4,32768:4,65536:4,1024:1,1024:8,1024:32"),
         "m26": ("google/gemma-4-26B-A4B-it", "2048:8,8192:8,32768:8,65536:8,65536:8:1,32768:8:1",
                 "1024:1,1024:8,1024:32,16384:16", "2048:8,32768:8,65536:8,1024:1,1024:8,1024:32")}


def stub(res: Path, key: str, *extra: str) -> None:
    model, cells, waste, prof = CELLS[key]
    subprocess.run([sys.executable, str(HERE / "decode_probe.py"), "--stub", "--model", model, "--out", str(res / key),
                    "--cells", cells, "--waste-cells", waste, "--profile-cells", prof, *extra],
                   check=True, capture_output=True)


def env(res: Path, copy: float | None = 3000.0, throttled: bool = False) -> None:
    e = res / "env-1"
    e.mkdir(parents=True, exist_ok=True)
    reason = "0x8" if throttled else "0x0"
    (e / "telemetry.csv").write_text(TEL + "".join(f"2026/10/30 10:00:00.000, 600, 1980, 60, 99, {reason}\n" for _ in range(150)))
    if copy is not None:
        (e / "microbench_start.json").write_text(json.dumps({"copy_gb_per_s_median": copy}))


def both(res: Path, *extra: str, copy: float | None = 3000.0) -> dict:
    for k in CELLS:
        stub(res, k, *extra)
    env(res, copy)
    return an.summarize(res)


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
    assert abs(f["b_low"] - 0.5) < 1e-9 and abs(f["b_up"] - 0.5) < 1e-9


def test_go(tmp_path: Path) -> None:
    s = both(tmp_path)
    assert all(m["valid"] for m in s["models"].values()), s["models"]["m31"].get("why")
    assert an.decide(s).startswith("GO (")
    m31 = s["models"]["m31"]
    assert 0.08 < m31["decide"]["32768"]["S"] < 0.11 and 0.12 < m31["decide"]["65536"]["S"] < 0.16   # byte model
    assert m31["fit"]["df"] == 4 and m31["V4c"] is True and m31["M2"] is True     # cell-level fit; flat sliding
    assert abs(m31["decide"]["65536"]["S_prof"] - m31["decide"]["65536"]["S"]) < 0.01
    assert m31["attribution"]["split"]["32768"]["groups"] == 8 * 60
    assert m31["kv_tokens_logged"] == 400000 and m31["max_concurrency_logged"] == 6.06
    assert 0 < m31["exploratory"]["16384:8"]["S_est"] < m31["decide"]["32768"]["S"] + 0.01


def test_kill_when_attention_is_tiny(tmp_path: Path) -> None:
    """The global attention is a normal-bandwidth kernel, but the rest of the step is 400 ms (S below 1%)."""
    s = both(tmp_path, "--stub-c-ms", "400", "--stub-noise", "0.0005")
    assert all(m["valid"] for m in s["models"].values()) and an.decide(s).startswith("KILL")


def test_pivot_kernel_bound(tmp_path: Path) -> None:
    d = an.decide(both(tmp_path, "--stub-bw-gbps", "800"))
    assert d.startswith("PIVOT") and "KERNEL-BOUND" in d


def test_pivot_attribution(tmp_path: Path) -> None:
    s = both(tmp_path, "--stub-bad-attribution")
    assert s["models"]["m31"]["V4b"] is False and s["models"]["m31"]["valid"]
    d = an.decide(s)
    assert d.startswith("PIVOT") and "attribution unclear" in d and "S_prof gap" in d


def test_pivot_attribution_unknown(tmp_path: Path) -> None:
    """R-64 R4: attention kernels that the regex does not match give 'unknown', not KERNEL-BOUND."""
    s = both(tmp_path, "--stub-opaque-names")
    m31 = s["models"]["m31"]
    assert m31["V4b"] is None and m31["V4c"] is None and m31["M1"]
    d = an.decide(s)
    assert d.startswith("PIVOT") and "attribution unknown" in d and "KERNEL-BOUND" not in d


def test_attention_names() -> None:
    """R-64 R4: real kernel names (FA3 names from the E-46 H100 profiles, Triton, vLLM) and their classes."""
    table = [
        ("void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90"
         "<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<112>, cute::C<256> >, "
         "256, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false>", True, "attention"),
        ("void flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<256> >, 7, 256, 1, false, true, "
         "cutlass::bfloat16_t, float, cutlass::arch::Sm90>(flash::FlashAttnFwdCombine::Params)", True, "attention"),
        ("void flash::prepare_varlen_num_blocks_kernel<1, true>(int, int, int, int const*)", True, "attention"),
        ("kernel_unified_attention_2d", True, "attention"),
        ("kernel_unified_attention_3d", True, "attention"),
        ("reduce_segments", True, "attention"),
        ("kernel_cutlass_kernel_flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__tensorptr", True,
         "attention"),   # FA4 (CuTe DSL) style; the real names come from the first H100 profile
        ("void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(long)",
         False, "copy_cache"),
        ("sm90_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x128_warpgroupsize1x1x1_execute_segment_k_off_kernel"
         "__5x_cublas", False, "gemm"),
        ("triton_poi_fused__to_copy_add_mul_rsqrt_0", False, "norm_act_rope"),
        ("void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel"
         "_cuda(at::TensorIteratorBase&)::{lambda()#3}> >(int)", False, "copy_cache"),
        ("fused_moe_kernel", False, "moe"),
        ("Memcpy DtoD (Device -> Device)", False, "copy_cache"),
    ]
    for name, attention, cls in table:
        assert an.is_attention(name) is attention, name
        got = next((c for c, rx in an.CLASSES if rx.search(name)), "other")
        assert got == cls, (name, got)


def test_profile_wall_does_not_break_v4a(tmp_path: Path) -> None:
    """R-64 R1: a large profiler start and stop cost (first profile 3x) must not change V4a or the idle share."""
    s = both(tmp_path, "--stub-profile-wall-ms", "400")
    m31 = s["models"]["m31"]
    assert m31["checks"]["V4a"] and m31["valid"] and an.decide(s).startswith("GO (")
    assert m31["waste"]["2048:4"]["profile_wall_ms_per_step"] > m31["waste"]["2048:4"]["wall_ms_per_step"] + 100
    assert all(0 <= w["idle_share"] < 0.5 for w in m31["waste"].values())


def test_sliding_growth_fails_v4c(tmp_path: Path) -> None:
    """R-64 R6: sliding-layer attention that grows with L fails the negative control."""
    s = both(tmp_path, "--stub-sliding-growth", "0.2")
    m31 = s["models"]["m31"]
    assert m31["V4c"] is False and m31["V4b"] is True and m31["attribution"]["sliding_change"]["32768"] > 0.05
    d = an.decide(s)
    assert d.startswith("PIVOT") and "V4c False" in d


def test_cell_level_bound(tmp_path: Path) -> None:
    """R-64 R5: with a cell effect, the cell-level bound is wider than the block-level bound."""
    s = both(tmp_path, "--stub-cell-sd", "0.03", "--stub-seed", "3")
    for m in s["models"].values():
        assert m["fit"]["df"] == 4 and m["fit_blocks"]["df"] == 16
        assert m["decide"]["65536"]["S_low"] < m["decide"]["65536"]["S_block_low"]


def test_positive_control(tmp_path: Path) -> None:
    """R-64 R7: an impossible bandwidth or no copy bandwidth is INVALID, not KERNEL-BOUND."""
    s = both(tmp_path / "fast", "--stub-bw-gbps", "5000")
    assert not s["models"]["m31"]["checks"]["V5"] and "V5" in an.decide(s)
    assert an.decide(both(tmp_path / "nocopy", copy=None)).startswith("INVALID (no copy bandwidth")


def test_preempted_cell_and_ok_slope(tmp_path: Path) -> None:
    """R-64 R2 and R8: a preempted cell is left out and listed; a run that stopped after the slope cells counts."""
    s = both(tmp_path / "pre", "--stub-preempt", "65536:4:1,65536:8:1")
    m31 = s["models"]["m31"]
    assert m31["valid"] and m31["preempted"] == ["65536:4:1"] and m31["fit"]["df"] == 3
    s = both(tmp_path / "slope", "--stub-stop-after-slope")
    assert s["models"]["m31"]["status"] == "ok_slope" and s["models"]["m31"]["valid"] and not s["models"]["m31"]["exploratory"]
    assert an.decide(s).startswith("GO (")


def test_text_only_required(tmp_path: Path) -> None:
    """R-64 R3: without the text-only log line the model is INVALID (V9)."""
    d = an.decide(both(tmp_path, "--stub-multimodal"))
    assert d.startswith("INVALID") and "V9" in d


def test_curvature_fails_m2(tmp_path: Path) -> None:
    """R-64 O1: a concave T(L) gives a local slope far below the fit; M2 blocks a pass."""
    s = both(tmp_path, "--stub-curve", "-0.3")
    m31 = s["models"]["m31"]
    assert m31["checks"]["V3"] and m31["M2"] is False and m31["local_slope_ratio"] < 0.75
    assert not an.decide(s).startswith("GO")


def test_invalid_paths(tmp_path: Path) -> None:
    for k in CELLS:
        stub(tmp_path / "acc", k, "--stub-accuracy", "0.2")
        stub(tmp_path / "adv", k, "--stub-bad-advance")
        stub(tmp_path / "tel", k)
    env(tmp_path / "acc")
    env(tmp_path / "adv")
    env(tmp_path / "tel", throttled=True)
    d = an.decide(an.summarize(tmp_path / "acc"))
    assert d.startswith("INVALID") and "V6" in d
    assert "V2" in an.decide(an.summarize(tmp_path / "adv"))
    assert an.decide(an.summarize(tmp_path / "tel")) == "INVALID (V8 telemetry)"


def test_one_model_and_noise(tmp_path: Path) -> None:
    stub(tmp_path / "one", "m31")
    env(tmp_path / "one")
    assert an.decide(an.summarize(tmp_path / "one")).startswith("GO (one model)")
    stub(tmp_path / "noisy", "m31", "--stub-noise", "0.05")
    stub(tmp_path / "noisy", "m26", "--stub-noise", "0.05")
    env(tmp_path / "noisy")
    s = an.summarize(tmp_path / "noisy")
    assert not s["models"]["m31"]["checks"]["V7"] or not s["models"]["m31"]["checks"]["V3"]


class FakeEngine:
    """Prefills each request in 3 steps, then adds 1 token per request per step; can preempt one request."""

    def __init__(self, preempt: bool = False) -> None:
        self.reqs: dict[str, dict] = {}
        self.preempt = preempt
        self.sched = types.SimpleNamespace(running=[], waiting=[])
        self.engine_core = types.SimpleNamespace(engine_core=types.SimpleNamespace(scheduler=self.sched))

    def add_request(self, rid: str, prompt: dict, params: dict) -> None:
        assert len(prompt["prompt_token_ids"]) == 1024 and params["ignore_eos"]
        self.reqs[rid] = {"left": 3, "tokens": 0, "pre": 0}

    def step(self) -> list:
        outs = []
        for rid, r in self.reqs.items():
            if r["left"]:
                r["left"] -= 1
            else:
                r["tokens"] += 1
            if r["tokens"]:
                outs.append(types.SimpleNamespace(request_id=rid, outputs=[types.SimpleNamespace(token_ids=[0] * r["tokens"])]))
        if self.preempt and self.reqs:
            next(iter(self.reqs.values()))["pre"] = 1
        self.sched.running = [types.SimpleNamespace(num_preemptions=r["pre"]) for r in self.reqs.values()]
        return outs

    def abort_request(self, ids: list[str]) -> None:
        for i in ids:
            self.reqs.pop(i, None)


def test_probe_loop_on_fake_engine(tmp_path: Path, monkeypatch) -> None:
    """R-64 O6: run_cell times blocks, records the advance and the scheduler check, skips a preempted cell and
    aborts every request."""
    import decode_probe as dp

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(dp, "sync", lambda: None)
    args = types.SimpleNamespace(steps=8, max_blocks=5, min_blocks=3, block_spread=10.0, profile_steps=2)
    stream = list(range(5000))
    eng = FakeEngine()
    cell, table = dp.run_cell(eng, stream, 1024, 4, 1, args, False, tmp_path)
    assert cell["status"] == "ok" and table is None and len(cell["step_ms"]) == 3 and cell["rep"] == 1
    assert cell["advance_min_max"] == [[8, 8]] * 3 and cell["sched"]["ok"] and cell["sched_after"]["ok"] and not eng.reqs
    eng = FakeEngine(preempt=True)
    cell, table = dp.run_cell(eng, stream, 1024, 4, 0, args, True, tmp_path)
    assert cell["status"] == "preempted" and cell["step_ms"] == [] and table is None and not eng.reqs
    assert cell["sched"]["preemptions"] == 1
    assert dp.cell_list("2048:4,65536:4:1") == [(2048, 4, 0), (65536, 4, 1)] and dp.cell_key(65536, 4, 1) == "65536:4:1"


def clean_copy(tmp_path: Path) -> Path:
    """R-64 R9: only the upload folders, without results."""
    code = tmp_path / "code"
    for sub in UPLOAD:
        shutil.copytree(HERE.parent / sub, code / sub, ignore=shutil.ignore_patterns("results", "dryrun", "__pycache__", "*.gz"))
    return code


def session(code: Path, res: Path, **extra: str) -> subprocess.CompletedProcess:
    e = dict(os.environ, STUB="1", RES=str(res), CODE=str(code), AUTO_STOP="0", **extra)
    return subprocess.run([shutil.which("sh"), str(code / "gemma4_kv_audit" / "session_e49.sh")], env=e,
                          capture_output=True, text=True, timeout=300)


def test_stub_session(tmp_path: Path) -> None:
    code, res = clean_copy(tmp_path), tmp_path / "e49"
    r = session(code, res)
    assert r.returncode == 0, r.stderr[-2000:]
    log = (res / "session.log").read_text()
    assert "m31: exit 0" in log and "m26: exit 0" in log and "decision: GO (" in log and "(one model)" not in log
    session(code, res)
    assert "m31: ok before, skip" in (res / "session.log").read_text()


def test_stub_session_failure_cap_and_rerun(tmp_path: Path) -> None:
    """PROTOCOL S3: a failed run and the time cap, then a rerun that keeps the failed output and finishes."""
    code, res = clean_copy(tmp_path), tmp_path / "e49"
    session(code, res, STUB_FAIL="m31", NEW_RUN_CAP="600")
    log = (res / "session.log").read_text()
    assert "m31: exit 1" in log and "time cap: skip m26" in log and "decision: INVALID" in log
    session(code, res)
    log = (res / "session.log").read_text()
    assert list(res.glob("m31.failed.*")) and log.rstrip().splitlines()[-2].split("decision: ")[1].startswith("GO (")

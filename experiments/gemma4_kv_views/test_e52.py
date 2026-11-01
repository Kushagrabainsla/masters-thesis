"""Tests for E-52 with amendment A1 (no GPU, no Triton): the views layout identity and its two-scale engine rule, the
Triton source of every layout and compute dtype on the CPU emulation (`cpu_dryrun_e52.py`), every decision path on
stub data (A1: the byte-route KILL, the flags, the baseline check), the R-70 and R-71 fixes of V3 and V4, the
deadline, the sticky-error stop, the compile-check skip, the diagnostics, and a stub session from a clean copy of the
upload folders.

Run: uv run --no-project --with numpy --with torch --with pytest python -m pytest test_e52.py
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))

import analyze_e52 as an  # noqa: E402
import compact_attn as ca  # noqa: E402
import kbench_e52 as kbm  # noqa: E402

UPLOAD = ["gemma4_kv_views", "gemma4_kv_kernel", "runpod", "fp8_quant_fusion", "nvfp4_dispatch", "safedispatch_testA",
          "safedispatch_testA2", "ledger"]   # README Section 11: the folders the Pod needs
TEL = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"
KB = [sys.executable, str(HERE / "kbench_e52.py"), "--stub"]
BEST = {"splits": 32, "warps": 4, "tile": 32, "stages": 2}       # the stub's fastest configuration


def views_case(seed: int = 3):
    gen = torch.Generator().manual_seed(seed)
    w = 1 + 0.1 * torch.randn(512, generator=gen, dtype=torch.float64)
    k, v = ca.make_kv(700, 4, w, gen, torch.float64)
    q = ca.rope(torch.randn(32, 512, generator=gen, dtype=torch.float64) / 512 ** 0.5,
                torch.full((32,), 699.0, dtype=torch.float64))
    return w, k, v, q


def test_views_identity() -> None:
    """The views row [K_rot | V_unrot | V_rot]: its first 512 values with the views query are the K, its last 512
    values are the V in the view order; the attention after `unpermute` is the standard attention (float64)."""
    w, k, v, q = views_case()
    row = ca.pack_views(k, v)
    kview, vview = ca.views_of(row)
    assert row.shape[-1] == 640 and kview.shape[-1] == vview.shape[-1] == 512 and len(ca.UNROT) == 384
    assert torch.equal(vview, v[..., ca.UNROT + ca.ROT])                       # the V view is V, reordered
    assert torch.equal(kview[..., 128:], vview[..., :384])                     # the two views share 384 values
    std = ca.gqa_attend(q, k, v)
    got = ca.unpermute(ca.gqa_attend(ca.views_query(q, w), kview, vview))
    assert float((std - got).norm() / std.norm()) < 1e-12
    assert torch.equal(ca.unpermute(v[..., ca.UNROT + ca.ROT]), v)


def test_two_scale_engine_rule() -> None:
    """A1 (R-71 O1): K_rot stored with scale s_k and V with s_v; one descale s_v for both views, and the rotated part
    of the query times s_k / s_v, give the exact attention."""
    w, k, v, q = views_case(4)
    s_k, s_v = 3.0, 0.25
    row = ca.pack_views(k / s_k, v / s_v)                  # the stored values (before an FP8 cast)
    kview, vview = ca.views_of(row)
    qv = ca.views_query(q, w)
    qv[..., :128] *= s_k / s_v
    got = ca.unpermute(ca.gqa_attend(qv * s_v, kview, vview) * s_v)      # k_descale = v_descale = s_v
    std = ca.gqa_attend(q, k, v)
    assert float((std - got).norm() / std.norm()) < 1e-12


def test_central_spread_ignores_one_outlier() -> None:
    times = [100.0 + 0.01 * i for i in range(29)] + [130.0]
    assert kbm.central_spread(times) < 0.003 and max(times) / min(times) - 1 > 0.29


def test_ptx_counts_with_predicates() -> None:
    """A1 (R-71 F6): predicated lines count too."""
    ptx = "\tcvt.rn.f16x2.e4m3x2 %r1, %rs1;\n\tcvt.bf16.f16 %rs2, %rs3;\n\tmma.sync.aligned.m16n8k16 ...;\n" \
          "\t@%p3 ld.shared.v4.b32 ...;\n\t@!%p12 cp.async.cg.shared.global ...;\n\tbar.sync 0;\n" \
          "\t@%p1 ld.global.v4.b32 ...;\n"
    c = kbm.ptx_counts(ptx)
    assert c["cvt.rn.f16x2.e4m3x2"] == c["cvt.bf16.f16"] == c["mma.sync"] == c["ld.shared"] == 1
    assert c["cp.async"] == c["bar.sync"] == c["ld.global"] == 1 and c["wgmma.mma_async"] == 0


def test_flags() -> None:
    """A1 (R-71 B1): the three named causes of a low sigma_layout."""
    base = {"sigma_layout": 0.05, "pool_share": {"triton_cmp": 0.6}}
    assert an.flags(32768, base | {"e_h": 0.3}) == ["a"]                      # time does not follow bytes
    assert an.flags(65536, base | {"e_h": 0.35}) == []                       # 0.35 >= 0.333 at 64K
    assert an.flags(32768, base | {"e_h": 0.9}) == ["b"]                      # 0.05 < 0.25 x 0.9: no sharing
    assert an.flags(32768, {"sigma_layout": 0.3, "e_h": 0.9, "pool_share": {"triton_cmp": 0.1}}) == ["c"]
    assert an.E_LINE == {32768: 0.4133, 65536: 0.3333}


def test_kernels_and_gpu_paths_on_the_cpu_emulation() -> None:
    """The Triton source (standard, views and fused layouts; BF16 and FP16 compute; FP8 and BF16 cache; partial
    pages with poisoned tail slots; empty splits) against the references; the views outputs against the exact
    attention; the K = V and pool steps; bench_cell and confirm."""
    r = subprocess.run([sys.executable, str(HERE / "cpu_dryrun_e52.py")], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["worst_kernel_err"] < 0.005                                # tolerance 0.01; E-51 saw 0.0021
    for case in out["cases"].values():
        for k, v in case.items():
            if " triton_" in k:
                assert v["all_float"] and v["n"] in (1, 2), (k, v)
            elif " stock_" in k:
                assert v == "error", k                                    # no vLLM on the Mac
        for name in ("fp8", "bf16"):
            e = [case[f"{name} exact_err_ref_{x}"] for x in ("std", "fused", "views")]
            assert max(e) - min(e) < 0.005, (name, e)                     # all three layouts store alike
            assert abs(case[f"{name} triton_cmpv_{'fp8h' if name == 'fp8' else 'bf16'}"]["exact_max"] - e[2]) < 0.005
    assert all(e < 0.005 for e in out["layout"].values()), out["layout"]
    assert out["info_keys"] == ["cubin", "ptx", "regs", "shared", "spills", "ttgir"]
    b = out["bench"]
    assert b["triton_ok"] == b["triton"] == 6 and b["stock_error"] == b["stock"] == 8 and b["has_regs_key"]
    assert all(v == 5 for k, v in out["confirm"].items() if k.startswith("triton")), out["confirm"]
    assert all(v == "error" for k, v in out["confirm"].items() if k.startswith("stock")), out["confirm"]


def stub_run(res: Path, *extra: str, copy: float | None = 3045.0, check: bool = True) -> None:
    kb = res / "kbench"
    kb.mkdir(parents=True, exist_ok=True)
    subprocess.run(KB + ["--out", str(kb), *extra], check=check, capture_output=True)
    env(res, copy)


def env(res: Path, copy: float | None = 3045.0, reasons: str = "0x0") -> None:
    e = res / "env-1"
    e.mkdir(exist_ok=True)
    (e / "telemetry.csv").write_text(TEL + "".join(f"2026/11/01 10:00:00.000, 600, 1980, 60, 99, {reasons}\n"
                                                   for _ in range(150)))
    if copy is not None:
        (e / "microbench_start.json").write_text(json.dumps({"copy_gb_per_s_median": copy}))


def analyze(res: Path) -> dict:
    s = an.summarize(res)
    s["decision"] = an.decide(s)
    return s


def stub(res: Path, *extra: str, copy: float | None = 3045.0, check: bool = True) -> dict:
    stub_run(res, *extra, copy=copy, check=check)
    return analyze(res)


def edit(res: Path, fn) -> None:
    p = res / "kbench" / "kbench.json"
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d))


def entry(d: dict, kernel: str, b: int, length: int, config: dict | None = None) -> dict:
    return next(e for e in d["confirm"] if e["kernel"] == kernel and e["batch"] == b and e["context"] == length
                and (config is None or e["config"] == config))


def test_go(tmp_path: Path) -> None:
    """The stub default is H4: FP16 compute makes the Triton pair bound by bytes, so the views layout gives 0.375."""
    s = stub(tmp_path)
    assert s["decision"].startswith("GO (replicated in 5 of 5 repeats;"), s["decision"]
    g = s["fp8"]["32768"]
    assert 0.36 < g["sigma_layout"] < 0.39 and g["sigma_mma"] > 0.4 and g["std_memory_bound"] and g["flags"] == []
    assert len(g["repeats"]) == 5 and g["range"]["sigma_mma"][0] > 0.25 and max(g["dt_range"].values()) < 0.03
    assert 0.15 < g["e_stock"] < 0.25 and 0.9 < g["e_h"] < 1.1 and 0.06 < g["sigma_stock_layout"] < 0.09
    assert g["pool_share"]["triton_std"] > 0.5 and g["pool_share"]["stock"] < 0.2 and g["e_bf16_triton"] > 0.9
    assert g["triton_std"]["config"] == g["triton_cmp"]["config"] == BEST and g["sigma_layout_own"] is not None
    assert set(g["cast_stage"]) == {f"{k} st{st}" for k in an.CAST_STAGE for st in (2, 3)}
    assert all(v for v in g["cast_stage"].values())
    assert all(s["checks"][k] for k in ("V1", "V2", "V3", "V4", "V5", "V6"))
    assert set(s["checks"]["_engine_ratio"]) == {"19:32768", "11:65536", "dt 19:32768", "dt 11:65536"}
    assert set(s["checks"]["_engine_ratio_2k"]) == {"19:2048", "11:2048"} and s["checks"]["_e51_stock_ratio"]
    assert s["checks"]["_cap"]["cap_share"] == 0 and s["checks"]["_rows_below_clock_flag"] == 0
    assert s["phases"]["confirm"] >= s["phases"]["correctness"] and s["lite_cells"] == []


def test_views_that_do_not_share_give_inconclusive(tmp_path: Path) -> None:
    """A1 (R-71 B1): a byte-bound twin and a views kernel that does not realize its byte saving give INCONCLUSIVE
    with flag (b), not KILL (the E-51 competence test would have said KILL here)."""
    s = stub(tmp_path, "--stub-onchip-fp16-ns", "0", "--stub-std-eff", "0.95", "--stub-cmp-eff", "0.62")
    assert s["decision"].startswith("INCONCLUSIVE (replicated") and "do not share" in s["decision"], s["decision"]
    assert s["fp8"]["32768"]["flags"] == ["b"] and s["fp8"]["32768"]["sigma_layout"] < 0.155


def test_byte_route_kill(tmp_path: Path) -> None:
    """A1 (R-71 B1, R-70 scope): H4 false, the pair stays bound on chip (e_h near 0) and the BF16 control shows that
    K = V halves the bytes: KILL of the byte route only."""
    s = stub(tmp_path, "--stub-onchip-fp16-ns", "0.435")
    assert s["decision"].startswith("KILL (replicated in 5 of 5 repeats; byte route only"), s["decision"]
    assert "stages V once is not tested" in s["decision"] and s["fp8"]["32768"]["e_h"] < 0.3


def test_byte_route_needs_a_valid_oracle(tmp_path: Path) -> None:
    """The same data with a BF16 K = V row that does not speed up (e_bf16_triton 0): INCONCLUSIVE, not KILL."""
    stub_run(tmp_path, "--stub-onchip-fp16-ns", "0.435")

    def no_bf16_gain(d):
        base = {(r["batch"], r["context"], json.dumps(r["config"], sort_keys=True)): r["us"] for r in d["cells"]
                if r["kernel"] == "triton_std_bf16"}
        for r in d["cells"]:
            if r["kernel"] == "triton_std_bf16_alias":
                r["us"] = base[(r["batch"], r["context"], json.dumps(r["config"], sort_keys=True))]

    edit(tmp_path, no_bf16_gain)
    s = analyze(tmp_path)
    assert s["decision"].startswith("INCONCLUSIVE (replicated") and "oracle is not valid" in s["decision"], \
        s["decision"]


def test_pivot_paths(tmp_path: Path) -> None:
    d = stub(tmp_path / "quality", "--stub-onchip-fp16-ns", "0", "--stub-std-eff", "0.55", "--stub-cmp-eff", "0.55")
    assert d["decision"].startswith("PIVOT") and "kernel quality" in d["decision"], d["decision"]
    d = stub(tmp_path / "bar", "--stub-onchip-fp16-ns", "0", "--stub-std-eff", "0.55", "--stub-cmp-eff", "0.42")
    assert d["decision"].startswith("PIVOT") and "below the bar" in d["decision"], d["decision"]


def test_invalid_and_inconclusive_paths(tmp_path: Path) -> None:
    d = stub(tmp_path / "engine", "--stub-engine-ratio", "1.3")["decision"]
    assert d.startswith("INVALID") and "V3" in d
    d = stub(tmp_path / "fast", "--stub-cmp-eff", "1.3")["decision"]
    assert d.startswith("INVALID") and "V5" in d
    assert stub(tmp_path / "nocopy", copy=None)["decision"].startswith("INVALID (no copy bandwidth")
    assert stub(tmp_path / "fail", "--stub-fail", check=False)["decision"].startswith("INCONCLUSIVE (the benchmark")
    d = stub(tmp_path / "slow", "--stub-onchip-fp16-ns", "0", "--stub-std-eff", "0.3", "--stub-cmp-eff", "0.3")
    assert d["decision"].startswith("INCONCLUSIVE") and "standard kernel reads below" in d["decision"]
    d = stub(tmp_path / "groups", "--cells", "19:2048,19:32768,11:2048")["decision"]     # no confirmation phase
    assert d.startswith("INVALID") or d.startswith("INCONCLUSIVE"), d


def test_baseline_off(tmp_path: Path) -> None:
    """A1 (R-71 F8): GO at 64K only because this session's stock dt is 4% above the E-51 value; with the E-51 stock
    dt the word is PIVOT, so the result is INCONCLUSIVE (baseline off)."""
    s = stub(tmp_path, "--stub-onchip-fp16-ns", "0", "--stub-std-eff", "0.927", "--stub-cmp-eff", "0.6736",
             "--stub-stock-eff", "0.6", "--stub-engine-ratio", "1.03", "--stub-repeat-noise", "0.0005")
    assert s["decision"].startswith("INCONCLUSIVE (baseline off: GO with this session's stock kernel, PIVOT"), \
        s["decision"]


def test_v3_bounds_the_x_cells_and_dt_only(tmp_path: Path) -> None:
    """R-70: a 2K time 20% below the engine leaves V3 (it is reported); the same at X fails V3."""
    stub_run(tmp_path)

    def low(length, f):
        def fn(d):
            for b in (19, 11):
                e = entry(d, "stock_fp8_s32", b, length if length != "X" else (32768 if b == 19 else 65536))
                e.update(repeats=[r * f for r in e["repeats"]], us=e["us"] * f)
        return fn

    edit(tmp_path, low(2048, 0.8))
    s = analyze(tmp_path)
    assert s["checks"]["V3"] and s["decision"].startswith("GO ("), s["decision"]
    assert abs(s["checks"]["_engine_ratio_2k"]["19:2048"] - 0.8) < 0.01
    edit(tmp_path, low("X", 0.8))
    assert "V3" in analyze(tmp_path)["decision"]


def test_v4_bounds_the_triton_rows_and_the_stock_dt_range(tmp_path: Path) -> None:
    """A1 (R-71 B2): a noisy stock row leaves V4 (reported); a noisy Triton decision row fails it; the stock dt range
    may reach 5%, a Triton dt range only 3%."""
    stub_run(tmp_path)
    stock = lambda d: next(e for e in d["confirm"] if e["kernel"].startswith("stock_fp8") and e["batch"] == 19
                           and e["context"] == 32768 and e["kernel"] in an.FAMILIES["stock"])
    edit(tmp_path, lambda d: stock(d).update(spreads=[0.06] * 5))
    s = analyze(tmp_path)
    assert s["decision"].startswith("GO (") and max(s["checks"]["_stock_spreads"]) == 0.06
    edit(tmp_path, lambda d: entry(d, "triton_cmpv_fp8h", 19, 32768, BEST).update(spreads=[0.06] * 5))
    assert analyze(tmp_path)["decision"] == "INVALID (V4)"
    edit(tmp_path, lambda d: entry(d, "triton_cmpv_fp8h", 19, 32768, BEST).update(spreads=[0.004] * 5))

    def drift(kernel_of, f):
        def fn(d):
            e = kernel_of(d)
            e["repeats"][2] *= f
        return fn

    edit(tmp_path, drift(stock, 1.035))                                   # stock dt range about 4%: allowed
    s = analyze(tmp_path)
    assert s["decision"].startswith("GO (") and 0.03 < s["checks"]["_stock_dt_range"] < 0.05, s["checks"]
    edit(tmp_path, drift(stock, 1.025))                                   # now about 6%
    assert analyze(tmp_path)["decision"] == "INVALID (V4)"
    stub_run(tmp_path / "t")
    edit(tmp_path / "t", drift(lambda d: entry(d, "triton_std_fp8h", 11, 65536, BEST), 1.05))
    s = analyze(tmp_path / "t")
    assert s["decision"] == "INVALID (V4)" and s["checks"]["_max_dt_range"] > 0.03


def test_v5_ignores_an_l2_resident_2k_row(tmp_path: Path) -> None:
    """E-51 A2, kept: a 2K row that reads from L2 faster than 1.15 x copy leaves V5."""
    stub_run(tmp_path)
    edit(tmp_path, lambda d: entry(d, "triton_cmpv_fp8h", 11, 2048, BEST).update(repeats=[15.0] * 5, us=15.0))
    s = analyze(tmp_path)
    assert s["checks"]["V5"] and s["decision"].startswith("GO ("), s["decision"]
    assert s["l2_ratios"]["2k"]["flag"]                                   # A1 (R-71 F1): the 11:2K views row flagged


def test_cap_share_is_reported(tmp_path: Path) -> None:
    """R-70: the software power cap (0x4) is counted and reported; it does not fail V6, a hard reason does."""
    stub_run(tmp_path)
    env(tmp_path, reasons="0x4")
    s = analyze(tmp_path)
    assert s["checks"]["V6"] and s["checks"]["_cap"]["cap_share"] == 1.0 and s["decision"].startswith("GO (")
    env(tmp_path, reasons="0x8")
    assert "V6" in analyze(tmp_path)["decision"]


def test_wrong_configuration_is_dropped(tmp_path: Path) -> None:
    """E-51 A1 (R5), kept: a views configuration that fails the correctness cell is never used. A1 (R-71 F1): the
    views kernel runs at the standard selection, so a failure there leaves the deciding family missing."""
    s = stub(tmp_path / "other", "--stub-bad-config", "64/4/32/2")
    assert s["decision"].startswith("GO (") and s["dropped_rows"] >= 4
    assert all(s["fp8"][X]["triton_cmp"]["config"] == BEST for X in ("32768", "65536"))
    s = stub(tmp_path / "selected", "--stub-bad-config", "32/4/32/2")
    assert s["decision"].startswith("INCONCLUSIVE (a deciding group is missing"), s["decision"]


def test_compile_check_crash_is_skipped(tmp_path: Path) -> None:
    """A1 (R-71 F2): an entry that crashed the compile check is skipped, so the next best configuration is used."""
    cc = tmp_path / "compile_check.json"
    cc.write_text(json.dumps({"kernels": {"triton_std_fp8h 32/4/32/2": {"status": "crashed"}}}))
    s = stub(tmp_path, "--compile-check", str(cc))
    assert s["decision"].startswith("GO ("), s["decision"]
    assert all(s["fp8"][X]["triton_std"]["config"] != BEST for X in ("32768", "65536"))


def test_fail_fast(tmp_path: Path) -> None:
    s = stub(tmp_path / "on", "--stub-bad-correct", check=False)
    assert s["decision"] == "INVALID (V2: fail fast, ['triton_cmpv_fp8h'] did not pass the correctness cell)"
    assert not json.loads((tmp_path / "on" / "kbench" / "kbench.json").read_text())["cells"]
    assert stub(tmp_path / "off", "--stub-bad-correct", "--fail-fast", "off")["decision"] == "INVALID (V2)"


def test_sticky_cuda_error_stops_the_run(tmp_path: Path) -> None:
    """A1 (R-71 O6): an illegal memory access stops the benchmark at once."""
    s = stub(tmp_path, "--stub-sticky", check=False)
    run = json.loads((tmp_path / "kbench" / "run.json").read_text())
    assert run["status"] == "error" and "illegal memory access" in run["error"]
    assert s["decision"].startswith("INCONCLUSIVE (the benchmark did not finish")


def test_deadline_runs_lite_rows(tmp_path: Path) -> None:
    """A1 (R-71 F4): after the deadline the group cells run the LITE rows only, and the decision still comes."""
    s = stub(tmp_path, "--deadline", "0")
    assert s["lite_cells"] == ["19:2048", "19:32768", "11:2048", "11:65536"] and s["decision"].startswith("GO (")
    n = sum(1 for r in json.loads((tmp_path / "kbench" / "kbench.json").read_text())["cells"]
            if r["kernel"] == "triton_std_fp8h" and r["batch"] == 19 and r["context"] == 32768)
    assert n == len(kbm.LITE)


def test_timeout_after_the_groups_keeps_the_decision(tmp_path: Path) -> None:
    kb = tmp_path / "kbench"
    kb.mkdir()
    p = subprocess.Popen(KB + ["--out", str(kb), "--stub-hang-after-groups", "60"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    run = kb / "run.json"
    for _ in range(300):
        if run.exists() and json.loads(run.read_text()).get("status") == "ok_groups":
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) != 0
    r = json.loads(run.read_text())
    assert r["status"] == "ok_groups" and "Timeout" in r["late_error"]
    env(tmp_path)
    s = analyze(tmp_path)
    assert s["decision"].startswith("GO (replicated"), s["decision"]
    assert s["batch4_us"]["triton_cmpv_fp8h"]["32768"] is None                    # the reported cells did not run


def test_decision_must_replicate(tmp_path: Path) -> None:
    """GO on the medians with sigma_mma just above 0.25; 2 of 5 repeats 1.5% slower fall below it (PIVOT). The dt
    range stays below 3%, so V4 passes and the replication test decides: INCONCLUSIVE."""
    stub_run(tmp_path, "--stub-onchip-fp16-ns", "0", "--stub-std-eff", "0.80", "--stub-cmp-eff", "0.6886",
             "--stub-repeat-noise", "0.0005")
    assert analyze(tmp_path)["decision"].startswith("GO (replicated in 5 of 5")

    def slow_two_repeats(d):
        for e in d["confirm"]:
            if e["kernel"] == "triton_cmpv_fp8h" and e["context"] > 2048:
                e["repeats"][0] *= 1.015
                e["repeats"][1] *= 1.015

    edit(tmp_path, slow_two_repeats)
    s = analyze(tmp_path)
    assert s["checks"]["V4"], s["checks"]["_max_dt_range"]
    assert s["decision"].startswith("INCONCLUSIVE (the decision does not replicate: GO on the medians, ['PIVOT', "
                                    "'PIVOT', 'GO'"), s["decision"]


def test_diagnostics(tmp_path: Path) -> None:
    """Reported only: BF16 with its K = V and pool controls, the cast x layout factorial, the L2 ratios, the
    linearity."""
    s = stub(tmp_path)
    for X in ("32768", "65536"):
        b = s["bf16"][X]
        assert b["ok"] and 0.9 < b["e_bf16_triton"] < 1.1 and 0.9 < b["e_bf16_stock"] < 1.1
        assert 0.3 < b["sigma_layout"] < 0.4 and b["sigma_layout_fused"] is not None and b["pool_share_bf16"] > 0.9
        f = s["factorial"][X]
        assert set(an.DIAG) <= set(f) and f["sigma_layout triton_cmpv_fp8h"] > 0.3
        assert abs(f["sigma_layout triton_cmpv_fp8"]) < 0.05 and f["sigma_layout triton_cmp_fp8"] < 0
    assert s["l2_ratios"]["b4_2k"]["views_over_std"] > 0 and not s["l2_ratios"]["2k"]["flag"]
    assert len(s["linearity"]) == 6 and s["batch4_us"]["stock_fp8_s32_views"]["2048"]
    assert s["compile_check"] is None and s["dram_probe"] is None and s["ptx_views_identical"] is None


def clean_copy(tmp_path: Path) -> Path:
    code = tmp_path / "code"
    for sub in UPLOAD:
        shutil.copytree(HERE.parent / sub, code / sub,
                        ignore=shutil.ignore_patterns("results", "dryrun", "__pycache__", "*.gz"))
    return code


def session(code: Path, res: Path, **extra: str) -> subprocess.CompletedProcess:
    e = dict(os.environ, STUB="1", RES=str(res), CODE=str(code), AUTO_STOP="0", **extra)
    return subprocess.run([shutil.which("sh"), str(code / "gemma4_kv_views" / "session_e52.sh")], env=e,
                          capture_output=True, text=True, timeout=300)


def test_stub_session(tmp_path: Path) -> None:
    code, res = clean_copy(tmp_path), tmp_path / "e52"
    r = session(code, res)
    assert r.returncode == 0, r.stderr[-2000:]
    log = (res / "session.log").read_text()
    assert "kbench: exit 0" in log and "decision: GO (replicated in 5 of 5" in log
    session(code, res)
    assert "kbench: ok before, skip" in (res / "session.log").read_text()
    session(code, tmp_path / "failed", STUB_ARGS="--stub-fail")
    assert "decision: INCONCLUSIVE" in (tmp_path / "failed" / "session.log").read_text()

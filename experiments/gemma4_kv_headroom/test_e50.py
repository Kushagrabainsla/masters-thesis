"""Tests for E-50 (no GPU): the token budget and its guard on a vLLM-like fake engine (the R-65 replay model), the
finish-reason record, the batch from the KV pool, every decision path on stub data, the count split, and stub
sessions (happy path, failed run, time cap, rerun) from a clean copy of the upload folders.

Run: uv run --no-project --with numpy --with pytest python -m pytest test_e50.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent / "gemma4_kv_audit" / "decode_probe.py"
sys.path.insert(0, str(PROBE.parent))

import analyze_e50 as an  # noqa: E402
import decode_probe as dp  # noqa: E402

UPLOAD = ["gemma4_kv_headroom", "gemma4_kv_audit", "safedispatch_testA2", "runpod", "fp8_quant_fusion",
          "nvfp4_dispatch", "safedispatch_testA", "ledger"]   # README Section 11: the folders the Pod needs
TEL = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"
G32 = "2048:a32768,8192:a32768,32768:a32768"
G64 = "2048:a65536,8192:a65536,32768:a65536,65536:a65536"
PP = "2048:4,32768:4,65536:4,2048:a32768,32768:a32768,2048:a65536,65536:a65536"
ARMS = {
    "m31f8": ("RedHatAI/gemma-4-31B-it-FP8-dynamic", "2048:4,8192:4,32768:4,65536:4,65536:4:1,32768:4:1",
              f"{G32},32768:a32768:1,{G64},65536:a65536:1", PP, "fp8"),
    "m31bf": ("RedHatAI/gemma-4-31B-it-FP8-dynamic", "2048:4,8192:4,32768:4,65536:4", f"{G32},{G64}", PP, "auto"),
    "m26": ("google/gemma-4-26B-A4B-it", "2048:8,8192:8,32768:8,65536:8,65536:8:1,32768:8:1", "",
            "2048:8,32768:8,65536:8", "auto"),
}
ARGS = dict(steps=32, max_blocks=5, min_blocks=3, block_spread=0.02, profile_steps=8, max_num_batched_tokens=2048,
            max_model_len=67584, auto_frac=0.85, max_num_seqs=32)


def stub(res: Path, key: str, *extra: str, check: bool = True) -> None:
    model, cells, pairs, prof, kv = ARMS[key]
    subprocess.run([sys.executable, str(PROBE), "--stub", "--model", model, "--out", str(res / key), "--cells", cells,
                    "--pair-cells", pairs, "--profile-cells", prof, "--kv-cache-dtype", kv, "--waste-cells", "",
                    "--max-model-len", "67584", *extra], check=check, capture_output=True)


def env(res: Path, copy: float | None = 3000.0) -> None:
    e = res / "env-1"
    e.mkdir(parents=True, exist_ok=True)
    (e / "telemetry.csv").write_text(TEL + "".join("2026/10/31 10:00:00.000, 600, 1980, 60, 99, 0x0\n" for _ in range(150)))
    if copy is not None:
        (e / "microbench_start.json").write_text(json.dumps({"copy_gb_per_s_median": copy}))


def all_arms(res: Path, *f8: str, bf: tuple = (), m26: tuple = ()) -> dict:
    stub(res, "m31f8", *f8)
    stub(res, "m31bf", *bf)
    stub(res, "m26", *m26)
    env(res)
    return an.summarize(res)


class BudgetEngine:
    """The R-65 replay model of the vLLM scheduler: a token budget of 2,048 per step, decode tokens first, FCFS
    prefill, a stop at max_tokens (finish reason "length"). stop_at = (id suffix, n) stops that request at n tokens
    with finish reason "stop". A request holds 0.0365 x prompt + 30 blocks of the pool."""

    def __init__(self, chunk: int = 2048, pool: int = 40000, stop_at: tuple[str, int] | None = None) -> None:
        self.chunk, self.pool, self.stop_at = chunk, pool, stop_at
        self.reqs: dict[str, dict] = {}
        pool_ns = types.SimpleNamespace(get_num_free_blocks=self.free)
        self.sched = types.SimpleNamespace(running=[], waiting=[], requests={},
                                           kv_cache_manager=types.SimpleNamespace(block_pool=pool_ns))
        self.engine_core = types.SimpleNamespace(engine_core=types.SimpleNamespace(scheduler=self.sched))

    def free(self) -> int:
        return self.pool - sum(r["blocks"] for r in self.reqs.values() if not r["done"])

    def _sync(self) -> None:
        live = {i: r for i, r in self.reqs.items() if not r["done"]}
        self.sched.running = [types.SimpleNamespace(num_preemptions=0) for r in live.values() if r["left"] == 0]
        self.sched.waiting = [i for i, r in live.items() if r["left"] > 0]
        self.sched.requests = {i: types.SimpleNamespace(num_preemptions=0) for i in live}

    def add_request(self, rid: str, prompt: dict, params: dict) -> None:
        n = len(prompt["prompt_token_ids"])
        self.reqs[rid] = {"left": n, "out": 0, "max": params["max_tokens"], "done": False, "blocks": int(0.0365 * n) + 30}
        self._sync()

    def step(self) -> list:
        budget = self.chunk
        dec = [i for i, r in self.reqs.items() if not r["done"] and r["left"] == 0 and r["out"] >= 1]
        budget -= len(dec)
        first = []
        for i, r in self.reqs.items():
            if not r["done"] and r["left"] > 0 and budget > 0:
                take = min(r["left"], budget)
                r["left"] -= take
                budget -= take
                if r["left"] == 0:
                    first.append(i)
        outs = []
        for i in dec + first:
            r = self.reqs[i]
            r["out"] += 1
            reason = "length" if r["out"] >= r["max"] else None
            if self.stop_at and i.endswith(self.stop_at[0]) and r["out"] >= self.stop_at[1]:
                reason = "stop"
            r["done"] = reason is not None
            outs.append(types.SimpleNamespace(request_id=i, finished=r["done"], outputs=[
                types.SimpleNamespace(token_ids=[0] * r["out"], finish_reason=reason, stop_reason=None)]))
        self._sync()
        return outs

    def abort_request(self, ids: list[str]) -> None:
        for i in ids:
            self.reqs.pop(i, None)
        self._sync()


def fake_vllm(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(dp, "sync", lambda: None)


def test_token_budget_and_cells() -> None:
    a = types.SimpleNamespace(**ARGS)
    assert dp.token_budget(65536, 8, a) == 7 * 33 + 4 + 160 + 8 + 32 == 435   # R-65: 435 for 64K at B = 8
    assert dp.token_budget(2048, 1, a) == 204
    assert dp.cell_list("2048:4,2048:a32768,32768:a32768:1") == [(2048, 4, 0), (2048, "a32768", 0), (32768, "a32768", 1)]
    assert dp.cell_key(32768, "a32768", 1) == "32768:a32768:1"
    assert dp.auto_batch("a32768", {32768: 1226.0}, 32190, a) == 22
    assert dp.auto_batch("a32768", {32768: 1226.0}, 10**6, a) == 32      # max_num_seqs
    assert dp.auto_batch("a65536", {32768: 1226.0}, 32190, a) is None   # no earlier cell of that context


def test_r65_case_old_budget_stops_new_budget_runs(tmp_path: Path, monkeypatch) -> None:
    """R-65 C3: the E-49 m26 64K cell (B = 8, chunk 2,048). The old budget (236) leaves 10 tokens; the guard now
    stops the cell before the timing. The new budget (435) keeps all 8 requests alive."""
    fake_vllm(monkeypatch)
    a, stream = types.SimpleNamespace(**ARGS), list(range(200_000))
    eng = BudgetEngine()
    cell, _ = dp.run_cell(eng, stream, 65536, 8, 0, a, False, tmp_path)
    assert cell["status"] == "ok" and cell["max_tokens"] == 435 and cell["tokens_at_start"] == 226
    assert cell["sched_after"]["running"] == 8 and cell["finished"] == {} and len(cell["step_ms"]) >= 3
    assert all(lo == 32 for lo, _ in cell["advance_min_max"]) and not eng.reqs
    assert cell["sched"]["request_preemptions"] == {f"c65536b8r0i{i}": 0 for i in range(8)}
    monkeypatch.setattr(dp, "token_budget", lambda *_: 236)
    cell, _ = dp.run_cell(BudgetEngine(), stream, 65536, 8, 0, a, False, tmp_path)
    assert cell["status"] == "budget" and cell["budget_left"] == 10 and cell["step_ms"] == []


def test_finish_reason_and_max_len(tmp_path: Path, monkeypatch) -> None:
    """A request that stops early is recorded with its reason and is not called a preemption (R-65)."""
    fake_vllm(monkeypatch)
    a, stream = types.SimpleNamespace(**ARGS), list(range(200_000))
    cell, _ = dp.run_cell(BudgetEngine(stop_at=("i0", 240)), stream, 65536, 8, 0, a, False, tmp_path)
    assert cell["status"] == "finished_early" and cell["finished"]["c65536b8r0i0"]["reason"] == "stop"
    assert cell["finished"]["c65536b8r0i0"]["tokens"] == 240
    cell, _ = dp.run_cell(BudgetEngine(stop_at=("i3", 50)), stream, 65536, 8, 0, a, False, tmp_path)
    assert cell["status"] == "finished_early" and cell["step_ms"] == []          # during the prefill of the others
    eng = BudgetEngine()
    cell, _ = dp.run_cell(eng, stream, 65536, 8, 0, types.SimpleNamespace(**(ARGS | {"max_model_len": 65836})), False,
                          tmp_path)
    assert cell["status"] == "budget" and not eng.reqs and cell["prefill_s"] == 0.0   # not run at all


def test_go(tmp_path: Path) -> None:
    s = all_arms(tmp_path)
    p = s["arms"]["m31f8"]
    assert p["valid"], p.get("why")
    assert an.decide(s).startswith("GO (")
    g32, g64 = p["groups"]["32768"], p["groups"]["65536"]
    assert (g32["batch"], g64["batch"]) == (22, 11) and g32["pool_share"] < 0.86       # 0.85 of the stub pool
    assert 0.10 < g32["S"] < 0.14 and 0.11 < g64["S"] < 0.16 and g64["M2"] is True and g32["usable"]
    assert 0.95 < g32["b_per_seq_ratio"] < 1.05   # the stub slope is linear in B, plus block noise (R-67 O4)
    assert p["decide"]["32768"]["S"] < g32["S"]                  # S grows with the batch
    assert not g32["S_att_gap"] and abs(g32["S_att"] - g32["S"]) < 0.01
    assert g32["attribution"]["split"]["32768"]["method"] == "order" and p["checks"]["V11"]
    bf = s["arms"]["m31bf"]
    assert bf["valid"] and (bf["groups"]["32768"]["batch"], bf["groups"]["65536"]["batch"]) == (11, 5)
    assert s["fp8_saving_B4"]["32768"] > 0.05 and bf["fit"]["n"] == 4


def test_kill(tmp_path: Path) -> None:
    s = all_arms(tmp_path, "--stub-c-ms", "2000", "--stub-noise", "0.0001")
    assert s["arms"]["m31f8"]["valid"] and an.decide(s).startswith("KILL")


def test_pivot_kernel_bound(tmp_path: Path) -> None:
    d = an.decide(all_arms(tmp_path, "--stub-bw-gbps", "1200"))
    assert d.startswith("PIVOT") and "KERNEL-BOUND" in d


def test_pivot_below_bar(tmp_path: Path) -> None:
    d = an.decide(all_arms(tmp_path, "--stub-c-ms", "60"))
    assert d.startswith("PIVOT") and "below the 5% bar" in d


def test_inconclusive_paths(tmp_path: Path) -> None:
    """R-65: a lost deciding group is INCONCLUSIVE, not PIVOT; a failed FP8 arm is INCONCLUSIVE."""
    s = all_arms(tmp_path / "lost", "--stub-preempt", "65536:a65536,65536:a65536:1", "--stub-c-ms", "60")
    assert s["arms"]["m31f8"]["groups"]["65536"]["ok"] is False
    assert an.decide(s).startswith("INCONCLUSIVE (deciding group lost: 65536")
    res = tmp_path / "fail"
    stub(res, "m31f8", "--stub-fail", check=False)
    stub(res, "m31bf")
    stub(res, "m26")
    env(res)
    assert an.decide(an.summarize(res)).startswith("INCONCLUSIVE (the FP8 arm did not finish")


def test_invalid_paths(tmp_path: Path) -> None:
    d = an.decide(all_arms(tmp_path / "v11", "--kv-cache-dtype", "auto"))   # the FP8 path did not run
    assert d.startswith("INVALID") and "V11" in d
    d = an.decide(all_arms(tmp_path / "v5", "--stub-bw-gbps", "5000"))
    assert d.startswith("INVALID") and "V5" in d
    res = tmp_path / "nocopy"
    for k in ARMS:
        stub(res, k)
    env(res, copy=None)
    assert an.decide(an.summarize(res)).startswith("INVALID (no copy bandwidth")


def test_m26_m2_unknown_without_64k(tmp_path: Path) -> None:
    """R-65 C3: with no 64K cell, M2 is unknown (E-49 counted it as true), so no m26 cell passes."""
    s = all_arms(tmp_path, m26=("--stub-preempt", "65536:8,65536:8:1"))
    m26 = s["arms"]["m26"]
    assert m26["local_slope_ratio"] is None and m26["M2"] is None and not any(m26["e49_pass"].values())


def test_pair_cells_stay_out_of_the_slope(tmp_path: Path) -> None:
    """A pair cell whose pool batch is 4 by chance is not a slope cell (the E-49 filter)."""
    s = all_arms(tmp_path, bf=("--stub-pool-blocks", "12000"))
    bf = s["arms"]["m31bf"]
    assert bf["groups"]["65536"]["batch"] == 4 and bf["fit"]["n"] == 4


def test_v11_needs_the_attention_path(tmp_path: Path) -> None:
    """A1 (R-67 R3): a Triton backend line or a Triton attention kernel fails V11, even with the Gemma4Config line."""
    s = all_arms(tmp_path, "--stub-triton")
    p = s["arms"]["m31f8"]
    assert not p["checks"]["V11"] and p["triton_kernels"] and "V11" in an.decide(s)


def test_group_validity_is_local(tmp_path: Path) -> None:
    """A1 (R-67 R4): a group whose own checks fail is lost, not INVALID. The other group can still decide."""
    s = all_arms(tmp_path / "spread", "--stub-spread-cells", "32768:a32768,32768:a32768:1")
    g32 = s["arms"]["m31f8"]["groups"]["32768"]
    assert g32["ok"] and not g32["usable"] and g32["why"] == ["V7"] and an.decide(s).startswith("GO (65536")
    s = all_arms(tmp_path / "primary", "--stub-preempt", "32768:a32768", "--stub-c-ms", "60")
    g32 = s["arms"]["m31f8"]["groups"]["32768"]
    assert g32["ok"] and "profiles" in g32["why"] and s["arms"]["m31f8"]["valid"]
    assert an.decide(s).startswith("INCONCLUSIVE (deciding group lost: 32768 (")


def test_ok_pairs_status_and_accuracy_caveat(tmp_path: Path) -> None:
    """A1 (R-67 O1, O2): a run.json left at "ok_pairs" is analyzed; low FP8 accuracy is a caveat, not INVALID."""
    s = all_arms(tmp_path / "pairs")
    run = tmp_path / "pairs" / "m31f8" / "run.json"
    run.write_text(run.read_text().replace('"status": "ok"', '"status": "ok_pairs"'))
    assert an.decide(an.summarize(tmp_path / "pairs")).startswith("GO (")
    s = all_arms(tmp_path / "acc", "--stub-accuracy", "0.2")
    d = an.decide(s)
    assert s["arms"]["m31f8"]["valid"] and d.startswith("GO (") and "accuracy caveat: GSM8K 0.2" in d


def test_kill_needs_the_pool_share(tmp_path: Path) -> None:
    """A1 (R-67 O6): a KILL needs B near the pool limit; at pool share 0.61 the same S gives INCONCLUSIVE."""
    s = all_arms(tmp_path, "--stub-c-ms", "2000", "--stub-noise", "0.0001", "--stub-pool-blocks", "2000")
    g = s["arms"]["m31f8"]["groups"]
    assert (g["32768"]["batch"], g["65536"]["batch"]) == (2, 1) and g["65536"]["pool_share"] < 0.75
    assert an.decide(s).startswith("INCONCLUSIVE (pool share below 0.75")


def test_run_plan_on_budget_engine(tmp_path: Path, monkeypatch) -> None:
    """A1 (R-67 O9): the real plan loop sizes the pair batch from the pool, marks a pair with no earlier cell of its
    context as no_size, and writes the "ok_pairs" status."""
    fake_vllm(monkeypatch)
    a = types.SimpleNamespace(**ARGS, profile_cells="")
    record: dict = {"status": "started"}
    cells, tables = dp.run_plan(BudgetEngine(), list(range(200_000)), [(2048, 4, 0), (65536, 4, 0)],
                                [(2048, "a65536", 0), (65536, "a65536", 0), (2048, "a32768", 0)], [], a, tmp_path, record)
    by = {c.get("spec") or f"{c['context']}:{c['batch']}": c for c in cells}
    assert record["free_blocks_idle"] == 40000 and record["status"] == "ok_pairs" and tables == {}
    assert by["65536:4"]["blocks_per_request"] == 2422.0 and by["65536:a65536"]["batch"] == int(0.85 * 40000 / 2422)
    assert by["65536:a65536"]["status"] == "ok" and by["65536:a65536"]["finished"] == {}
    assert by["2048:a32768"]["status"] == "no_size" and (tmp_path / "cells.json").exists()


def test_count_split() -> None:
    """With FA3 on the sliding layers and FA4 on the global layers, the launch-order split may fail; the count split
    assigns each attention kernel by its launches per step (50 sliding, 10 global layers in the 31B model)."""
    types_ = (["sliding_attention"] * 5 + ["full_attention"]) * 10
    t = {"steps": 8, "kernels": [
        {"name": "void flash::FlashAttnFwdSm90<256>", "count": 400, "device_us": 800.0},
        {"name": "kernel_cutlass_flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_fp8", "count": 80, "device_us": 8000.0},
        {"name": "void flash::prepare_varlen_num_blocks_kernel", "count": 8, "device_us": 30.0},
        {"name": "sm90_xmma_gemm_e4m3", "count": 480, "device_us": 90000.0}]}
    s = an.split_of(Path("."), t, types_)
    assert s["ok"] and s["method"] == "count" and s["sliding_ms"] == 0.1 and s["global_ms"] == 1.0
    t["kernels"].append({"name": "void flash::FlashAttnFwdCombine", "count": 37, "device_us": 2000.0})
    assert not an.split_of(Path("."), t, types_)["ok"]


def clean_copy(tmp_path: Path) -> Path:
    """R-64 R9: only the upload folders, without results."""
    code = tmp_path / "code"
    for sub in UPLOAD:
        shutil.copytree(HERE.parent / sub, code / sub, ignore=shutil.ignore_patterns("results", "dryrun", "__pycache__", "*.gz"))
    return code


def session(code: Path, res: Path, **extra: str) -> subprocess.CompletedProcess:
    e = dict(os.environ, STUB="1", RES=str(res), CODE=str(code), AUTO_STOP="0", **extra)
    return subprocess.run([shutil.which("sh"), str(code / "gemma4_kv_headroom" / "session_e50.sh")], env=e,
                          capture_output=True, text=True, timeout=300)


def test_stub_session(tmp_path: Path) -> None:
    code, res = clean_copy(tmp_path), tmp_path / "e50"
    r = session(code, res)
    assert r.returncode == 0, r.stderr[-2000:]
    log = (res / "session.log").read_text()
    assert all(f"{k}: exit 0" in log for k in ARMS) and "decision: GO (" in log
    session(code, res)
    assert "m31f8: ok before, skip" in (res / "session.log").read_text()


def test_stub_session_failure_cap_and_rerun(tmp_path: Path) -> None:
    """PROTOCOL S3: a failed run and the time cap, then a rerun that keeps the failed output and finishes."""
    code, res = clean_copy(tmp_path), tmp_path / "e50"
    session(code, res, STUB_FAIL="m31f8", NEW_RUN_CAP="300")
    log = (res / "session.log").read_text()
    assert "m31f8: exit 1" in log and "time cap: skip m26" in log and "decision: INCONCLUSIVE" in log
    session(code, res)
    log = (res / "session.log").read_text()
    assert list(res.glob("m31f8.failed.*")) and log.rstrip().splitlines()[-2].split("decision: ")[1].startswith("GO (")

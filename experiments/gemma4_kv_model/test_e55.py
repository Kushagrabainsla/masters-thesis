"""E-55 tests (README Section 12, amendment A1). Run: uv run --no-project --with numpy --with torch --with pytest python
-B -m pytest -q -p no:cacheprovider experiments/gemma4_kv_model/test_e55.py"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_accuracy"), str(EXP / "gemma4_kv_kernel"), str(EXP / "ledger")]
import analyze_e55 as an55  # noqa: E402
import run_e55 as run  # noqa: E402
import splice_e55 as sp  # noqa: E402

CAP = EXP / "gemma4_kv_scaletie/results/e54-h100-20261102/capture"
SCALES = json.loads((EXP / "gemma4_kv_scaletie/scales_e54.json").read_text())["layers"]
DATA = HERE / "data_e55.json"


def attend(q, k, v, qpos):
    import analyze_e53 as an53
    return an53.attend(q, k, v, qpos)


def test_rot_dims_match_the_compact_layout():
    import compact_attn as ca
    rot, unrot = sp.rot_dims(512)
    assert rot == ca.ROT and unrot == ca.UNROT


def test_arm_lists_agree():
    assert an55.HOOKED == sp.ARMS and an55.DECISION == run.DECISION_ARMS
    assert set(run.REPORTED) - {"GEN"} == set(sp.ARMS) - set(run.DECISION_ARMS)


@pytest.mark.skipif(not (CAP / "p0/L05.pt").exists(), reason="the E-54 captures are not on this machine")
def test_transforms_reproduce_the_e54_errors_on_real_captures():
    """The splice numerics equal the frozen E-54 analysis: BF16C, STOCK8, TIED8, JOINT8 and TIED8_UP give the E-54
    errors of prompt 0 (layers 5 and 41) to 1e-4 relative, and every V5 check passes (BF16R included)."""
    import analyze_e53 as an53
    met = json.loads((CAP.parent / "e54/metrics.json").read_text())["prompts"]["0"]
    rot, unrot = sp.rot_dims(512)
    names = {"BF16C": "BF16C", "STOCK8": "STOCK", "TIED8": "TIED", "JOINT8": "CALSTD_JOINT", "TIED8_UP": "TIED_UP"}
    for layer in (5, 41):
        cap = torch.load(CAP / f"p0/L{layer:02d}.pt", map_location="cpu", weights_only=True)
        q, k, v, qpos = cap["q"], cap["k"], cap["v"], cap["qpos"].long()
        ref = an53.attend(q.float(), k.float(), v.float(), qpos)
        sc = SCALES[str(layer)]
        for arm in sp.ARMS:
            s = sp.arm_scales(arm, sc)
            q2, k2, v2 = sp.transform(arm, q, k, v, sc["w"], s, rot, unrot)
            assert sp.check_transform(arm, q, k, v, q2, k2, v2, sc["w"], s, rot, unrot) == []
            if arm in names:
                err = an53.rel(sp.emulate(q2, k2, v2, qpos, s, arm, an53.attend), ref)
                assert abs(err / met[str(layer)]["err"][names[arm]] - 1) < 1e-4, (layer, arm)


def toy_qkv(t=24, hq=4, hk=2, d=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(t, hk, d, generator=g).to(torch.bfloat16)
    w = 0.0615234375
    k = (v.float() * w).to(torch.bfloat16)
    rot, _ = sp.rot_dims(d)
    k[..., rot] = torch.randn(t, hk, len(rot), generator=g).to(torch.bfloat16) * 0.1
    q = (torch.randn(t, hq, d, generator=g) * 3).to(torch.bfloat16)
    return q, k, v, w


TOY_SC = {"w": 0.0615234375, "tied": {"vs": 0.125, "ks": sp.f32(0.0615234375 * 0.125)},
          "calstd_joint": {"ks": 0.01, "vs": 0.2}}


def test_check_transform_catches_planted_faults():
    q, k, v, w = toy_qkv()
    rot, unrot = sp.rot_dims(512)
    for arm in sp.ARMS:
        s = sp.arm_scales(arm, TOY_SC)
        q2, k2, v2 = sp.transform(arm, q, k, v, w, s, rot, unrot)
        assert sp.check_transform(arm, q, k, v, q2, k2, v2, w, s, rot, unrot) == [], arm
    s = sp.arm_scales("TIED8", TOY_SC)
    q2, k2, v2 = sp.transform("TIED8", q, k, v, w, s, rot, unrot)
    bad_k = k2.clone()
    bad_k[..., unrot[0]] = sp.fp8(k.float()[..., unrot[0]] / s["ks"]).to(k.dtype)     # the standard K code
    assert any("tie" in b for b in sp.check_transform("TIED8", q, k, v, q2, bad_k, v2, w, s, rot, unrot))
    q2, k2, v2 = sp.transform("BF16C", q, k, v, w, s, rot, unrot)
    assert sp.check_transform("BF16C", q, k, v, q, k2, v2, w, s, rot, unrot)          # the query not scaled
    q2, k2, v2 = sp.transform("STOCK8", q, k, v, w, s, rot, unrot)
    assert sp.check_transform("STOCK8", q, k, v, q2, k, v2, w, s, rot, unrot)          # K not cast
    s = sp.arm_scales("BF16R", TOY_SC)
    q2, k2, v2 = sp.transform("BF16R", q, k, v, w, s, rot, unrot)
    assert sp.check_transform("BF16R", q, k, v, q, k2, v2, w, s, rot, unrot)           # the query not scaled


class ToyImpl:
    def __init__(self):
        self.scale = 1.0


class ToyAttention(torch.nn.Module):
    """A stand-in of the vLLM Attention module: causal GQA attention of [T, H x D] inputs with impl.scale on the
    scores, returning [T, Hq x D] (as attention.py)."""

    def __init__(self, hq, hk, d):
        super().__init__()
        self.impl, self.num_heads, self.num_kv_heads, self.head_size = ToyImpl(), hq, hk, d

    def forward(self, query, key, value):
        t = key.shape[0]
        q = query.view(t, self.num_heads, self.head_size).float()
        k = key.view(t, self.num_kv_heads, self.head_size).float() * self.impl.scale
        v = value.view(t, self.num_kv_heads, self.head_size).float()
        return attend(q, k, v, torch.arange(t)).to(query.dtype).reshape(t, -1)


def test_splice_hooks_apply_each_arm_and_detach_cleanly():
    """On a toy module the hooks give the emulated output of every arm (the score factor through impl.scale, the V
    descale on the output), count the tokens, record both references, pass V5, and leave no trace after detach."""
    q, k, v, w = toy_qkv()
    t, hq, d = q.shape
    hk = k.shape[1]
    mod = ToyAttention(hq, hk, d)
    base = mod(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1))
    splice = sp.Splice({7: w}, {7: TOY_SC}, {7: (hq, hk, d)})
    splice.attach(7, mod)
    rot, unrot = sp.rot_dims(d)
    for arm in sp.ARMS:
        splice.set_arm(arm)
        splice.new_doc("X")
        got = mod(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)) if arm != "JOINT8" else \
            mod(query=q.reshape(t, -1), key=k.reshape(t, -1), value=v.reshape(t, -1))   # keyword call
        s = sp.arm_scales(arm, TOY_SC)
        q2, k2, v2 = sp.transform(arm, q, k, v, w, s, rot, unrot)
        want = sp.emulate(q2, k2, v2, torch.arange(t), s, arm, attend).to(torch.bfloat16).reshape(t, -1)
        if arm in sp.SCALED:      # the toy rounds the output to BF16 before the V descale, as the engine does
            want = (attend(q2.float(), k2.float() * s["ks"], v2.float(), torch.arange(t)).to(torch.bfloat16).float()
                    * s["vs"]).to(torch.bfloat16).reshape(t, -1)
        assert torch.equal(got, want), arm
        assert splice.counts[(arm, "X", 7)] == t
        if arm == "BF16":
            assert torch.equal(got, base)
    assert all(v == [] for v in splice.checks.values()) and splice.errors == {}
    assert set(splice.gaps) == set(splice.gaps_id) == set(splice.seps) == set(sp.ARMS)
    assert all(g[7] < 0.01 for g in splice.gaps.values()) and splice.seps["BF16"][7] == 0.0
    splice.set_arm("TIED8")
    saved = splice.detach()
    assert mod.impl.scale == 1.0
    assert torch.equal(mod(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)), base)
    splice.reattach(saved)
    splice.set_arm("STOCK8")
    assert not torch.equal(mod(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)), base)


class BrokenAttention(ToyAttention):
    """A kernel that ignores impl.scale (a backend that read the scale once at start)."""

    def forward(self, query, key, value):
        scale, self.impl.scale = self.impl.scale, 1.0
        try:
            return super().forward(query, key, value)
        finally:
            self.impl.scale = scale


def test_v5_gap_catches_a_kernel_that_ignores_the_k_descale():
    q, k, v, w = toy_qkv()
    t, hq, d = q.shape
    mod = BrokenAttention(hq, k.shape[1], d)
    splice = sp.Splice({7: w}, {7: TOY_SC}, {7: (hq, k.shape[1], d)})
    splice.attach(7, mod)
    for arm in ("STOCK8", "TIED8"):
        splice.set_arm(arm)
        splice.new_doc("X")
        mod(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1))
    assert splice.gaps["STOCK8"][7] < 0.01 and splice.gaps["TIED8"][7] > 0.3


class IgnoringAttention(ToyAttention):
    """A kernel that uses the inputs from before the splice hook (an engine that bypassed the hook's tensors)."""

    def __init__(self, hq, hk, d):
        super().__init__(hq, hk, d)
        self.orig = None
        self.register_forward_pre_hook(lambda m, a: setattr(self, "orig", a))

    def forward(self, query, key, value):
        return super().forward(*self.orig[:3])


def test_v7_catches_an_engine_that_ignores_the_hook():
    """R-77 F8: V5 alone cannot tell a hooked BF16C from one that the engine ignored (both gaps are below 0.01). V7
    asks that the engine output be nearer the arm's reference than the unchanged reference."""
    q, k, v, w = toy_qkv(t=64, seed=3)
    t, hq, d = q.shape
    for cls, broken in ((ToyAttention, False), (IgnoringAttention, True)):
        mod = cls(hq, k.shape[1], d)
        splice = sp.Splice({7: w}, {7: TOY_SC}, {7: (hq, k.shape[1], d)})
        splice.attach(7, mod)
        for arm in sp.ARMS:
            splice.set_arm(arm)
            splice.new_doc("X")
            mod(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1))
        ev = splice.evidence()
        v5 = {arm: an55.engine_problems(arm, 1, ev)[0] for arm in sp.ARMS}
        v7 = {arm: an55.engine_problems(arm, 1, ev)[1] for arm in sp.ARMS}
        assert ev["seps"]["BF16C"]["7"] > an55.SEP_MIN and ev["gaps"]["BF16C"]["7"] < an55.GAP_TOL
        if broken:   # V5 passes for BF16C (gap below 0.01); only V7 sees it. The scaled arms fail V5 (gap near 1).
            assert v5["BF16C"] == [] and v7["BF16C"] and all(v5[a] or v7[a] for a in sp.ARMS[1:]), (v5, v7)
        else:
            assert not any(v5.values()) and not any(v7.values()), (v5, v7)


def stub_splice(w_layers: dict[int, float]) -> tuple:
    engine = run.StubEngine(w_layers, "")
    splice = sp.Splice({}, {int(k): x for k, x in SCALES.items()}, {})
    for layer, m in engine.modules.items():
        splice.w[layer], splice.heads[layer] = w_layers[layer], run.STUB_HEADS
        splice.attach(layer, m)
    return engine, splice


def test_engine_order_gives_evidence_for_every_arm():
    """R-77 B1: the warm-up call (arm BF16, no new_doc) used the first check of each layer, so BF16 had no V5 gap and
    every real run would end INVALID. reset_evidence after the warm-up gives every arm its gaps in every layer."""
    w_layers = {layer: SCALES[str(layer)]["w"] for layer in run.GLOBAL_LAYERS}
    for reset in (False, True):
        engine, splice = stub_splice(w_layers)
        engine.prompt(run.WARM_TOKENS, 1)
        if reset:
            splice.reset_evidence()
        for arm in sp.ARMS:
            splice.set_arm(arm)
            splice.new_doc("B1")
            engine.prompt(run.PRE_TOKENS, 2)
        if reset:
            assert {a: len(g) for a, g in splice.gaps.items()} == {a: 10 for a in sp.ARMS}
        else:
            assert "BF16" not in splice.gaps and len(splice.gaps) == len(sp.ARMS) - 1


def flat(ids, lps, top1, top1_lp, ranks):
    """A FlatLogprobs as vLLM 3966924d5 builds it (logprobs.py 165 to 209)."""
    st, en, tok, lp, rk = [0], [0], [], [], []
    for i in range(1, len(ids)):
        st.append(len(tok))
        tok += [ids[i], top1[i - 1]]
        lp += [lps[i - 1], top1_lp[i - 1]]
        rk += [ranks[i - 1], 1]
        en.append(len(tok))
    return SimpleNamespace(start_indices=st, end_indices=en, token_ids=tok, logprobs=lp, ranks=rk)


def test_parse_prompt_logprobs_reads_the_vllm_layout_and_refuses_others():
    ids = [2, 10, 11, 12, 13]
    lps, top1, top1_lp, ranks = [-1.0, -2.0, -0.5, -3.0], [10, 99, 12, 98], [-1.0, -0.1, -0.5, -0.2], [1, 4, 1, 7]
    lp, t1, rk = run.parse_prompt_logprobs(flat(ids, lps, top1, top1_lp, ranks), ids)
    assert lp.tolist() == lps and t1.tolist() == top1 and rk.tolist() == ranks
    with pytest.raises(ValueError):
        run.parse_prompt_logprobs(flat(ids, lps, top1, top1_lp, ranks), [2, 10, 11, 12, 14])
    bad = flat(ids, lps, top1, top1_lp, ranks)
    bad.end_indices[-1] += 1
    with pytest.raises(ValueError):
        run.parse_prompt_logprobs(bad, ids)


def stats(bf16c=(0.0, -1e-5, 1e-5), stock=0.002, tied=0.001, d_ci=(-0.0012, -0.0008), e_ci=None):
    e = tied - an55.KILL8_RATIO * stock
    return {"delta": {"BF16C": bf16c[0], "E5M2": 0.005, "STOCK8": stock, "TIED8": tied},
            "ci": {"BF16C": (bf16c[1], bf16c[2])}, "m16": an55.M16, "D": tied - stock, "ci_D": d_ci,
            "tol8": max(an55.TOL8_ABS, an55.TOL8_REL * stock), "E": e, "ci_E": e_ci or (e - 0.0003, e + 0.0003)}


def test_decision_rule_gives_each_outcome():
    assert an55.decide16(stats())[0] == "GO16"
    assert an55.decide16(stats(bf16c=(0.02, 0.019, 0.021)))[0] == "KILL16"
    assert an55.decide16(stats(bf16c=(0.003, 0.002, 0.004)))[0] == "PIVOT16"
    assert an55.decide16(stats(bf16c=(0.0006, -0.0001, 0.0013)))[0] == "INCONCLUSIVE16"
    assert an55.decide16(stats(bf16c=(0.0004, -0.0001, 0.0009)))[0] == "GO16"          # the fixed margin 0.001
    assert an55.decide16(stats(bf16c=(-0.002, -0.003, -0.0015)))[0] == "CHECK16"       # an improvement: an artifact
    assert an55.decide8(stats())[0] == "GO8"
    assert an55.decide8(stats(tied=0.0021, d_ci=(-0.0002, 0.0004)))[0] == "GO8"        # tolerance 0.0005
    assert an55.decide8(stats(tied=0.004, d_ci=(0.0015, 0.0025)))[0] == "KILL8"        # E 0.001 [0.0007, 0.0013]
    assert an55.decide8(stats(tied=0.0028, d_ci=(0.0006, 0.001)))[0] == "PIVOT8"       # E below 0
    assert an55.decide8(stats(tied=0.0034, d_ci=(0.001, 0.0018), e_ci=(-0.0001, 0.0005)))[0] == "PIVOT8"
    assert an55.decide8(stats(tied=0.0024, d_ci=(-0.0001, 0.0009)))[0] == "INCONCLUSIVE8"


def test_t_quantile_matches_the_t_table():
    for p, df, want in ((0.975, 9, 2.262157), (0.975, 3, 3.182446), (0.975, 30, 2.042272), (0.95, 5, 2.015048),
                        (0.975, 1e9, 1.959964)):
        assert abs(an55.t_quantile(p, df) - want) < 1e-5, (p, df)
    assert an55.t_quantile(0.975, 4) > an55.t_quantile(0.975, 4.5) > an55.t_quantile(0.975, 5)   # Satterthwaite df


def test_t_interval_covers_where_the_percentile_bootstrap_does_not():
    """R-77 F5: with 4 documents per domain the percentile bootstrap is too narrow; the stratified t interval keeps
    about 95% coverage (here: unequal spreads by domain, the plan's token counts)."""
    rng = np.random.default_rng(7)
    doms = ["books"] * 4 + ["prose"] * 4 + ["code"] * 4
    n = np.array([64_511.0] * 4 + [31_743.0] * 8)
    mu = np.array([0.0002] * 4 + [-0.0001] * 4 + [0.0004] * 4)
    tau = np.array([3e-4] * 4 + [2e-4] * 4 + [5e-4] * 4)
    true = float((n * mu).sum() / n.sum())
    idx = an55.draws(doms, boot=1_000)
    hit_t = hit_b = 0
    sims = 1_500
    for _ in range(sims):
        s = n * (mu + tau * rng.standard_normal(12))
        lo, hi = an55.t_interval(s, n, doms)
        hit_t += lo <= true <= hi
        lo, hi = an55.interval(s, n, idx)
        hit_b += lo <= true <= hi
    assert 0.93 <= hit_t / sims <= 0.975, hit_t / sims
    assert hit_b / sims < hit_t / sims - 0.03, (hit_b / sims, hit_t / sims)


def test_bootstrap_is_stratified_and_fixed_by_the_seed():
    doms = ["books"] * 4 + ["prose"] * 4 + ["code"] * 4
    a, b = an55.draws(doms, boot=500), an55.draws(doms, boot=500)
    assert a.shape == (500, 12) and np.array_equal(a, b)
    for j, dom in enumerate(["books"] * 4 + ["prose"] * 4 + ["code"] * 4):
        assert all(doms[i] == dom for i in a[:, j])


def test_far_repeat_mask_finds_only_copies_beyond_the_window():
    rng = np.random.default_rng(0)
    ids = rng.integers(1_000, 200_000, 5_000).tolist()
    ids[3000:3012] = ids[100:112]            # a copy of a 12-gram that ended 2,900 tokens before
    ids[4000:4012] = ids[3500:3512]          # a copy inside the sliding window (500 tokens)
    m = an55.far_repeat_mask(ids)
    assert m[3011] and not m[4011] and m.sum() == 1


def test_data_file_is_the_frozen_plan():
    data = json.loads(DATA.read_text())
    assert an55.sha_file(DATA) == an55.DATA_SHA256
    docs = data["docs"]
    assert [d["id"] for d in docs] == ["B1", "B2", "B3", "B4", "P1", "P2", "P3", "P4", "C1", "C2", "C3", "C4"]
    names = [s["name"] for d in docs for s in d["sources"]] + [g["source"] for g in data["gen"]]
    assert len(names) == len(set(names))
    assert not any("gemma4_kv_model" in n for n in names)
    bos = data["bos_id"]
    for d in docs:
        want = 65_536 if d["domain"] == "books" else 32_768
        assert d["length"] == len(d["ids"]) == want and d["ids"][0] == bos and bos not in d["ids"][1:]
        assert run.ids_sha(d["ids"]) == d["ids_sha256"]
    assert len(data["gen"]) == 16
    for g in data["gen"]:
        assert g["ids"][0] == bos and g["ids"].count(bos) == 1 and run.ids_sha(g["ids"]) == g["ids_sha256"]
        assert g["ids"][-6:-4] == [4368, 107]                  # "model", newline: the generation prompt


def test_session_limits_fit_the_cost_cap():
    """The session watchdog fires before the boot watchdog, and the worst case (15 min to SSH, after which the run
    script deletes the Pod, then the boot watchdog) stays inside the cap of $4.00 at $3.99/h (A1, R-77 F9)."""
    text = (HERE / "session_e55.sh").read_text()
    limit = int(re.search(r"POD_LIMIT:-(\d+)", text).group(1))
    boot = int(re.search(r"BOOT_CAP:-(\d+)", text).group(1))
    assert limit + 150 < boot
    assert (900 + boot) / 3600 * 3.99 <= 4.00
    assert int(re.search(r"AUTO_STOP_DELAY:-(\d+)", text).group(1)) >= 300


def run_stub(tmp: Path, args: str = "", extra_env: dict | None = None) -> dict:
    res = tmp / "res"
    env = dict(os.environ, STUB="1", RES=str(res), CODE=str(EXP), STUB_ARGS=args, AUTO_STOP="0",
               SCALES=str(EXP / "gemma4_kv_scaletie/scales_e54.json"), POD_LIMIT="900",
               PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    env.update(extra_env or {})
    subprocess.run(["sh", str(HERE / "session_e55.sh")], env=env, check=True, capture_output=True, timeout=600)
    return json.loads((res / "summary.json").read_text())


def test_stub_session_runs_end_to_end(tmp_path):
    """The stub engine carries the real hooks in the engine order: every arm has its gaps in every layer (B1), BF16C
    decides V7 in the 9 layers whose w is not a power of 2, and the frozen rule decides."""
    s = run_stub(tmp_path)
    assert s["decision"] == "GO16 GO8", s.get("validity")
    assert set(s["stats"]["delta"]) == set(an55.SCORED)
    assert not any(s["validity"].values()) and s["excluded"] == {} and s["preflight"] == []
    rec = json.loads((tmp_path / "res/run_main.json").read_text())
    assert {a: len(g) for a, g in rec["gaps"].items()} == {a: 10 for a in sp.ARMS}
    assert sum(x > an55.SEP_MIN for x in rec["seps"]["BF16C"].values()) == 9 and rec["seps"]["BF16C"]["29"] == 0.0
    assert s["stats"]["gen"]["BF16C"]["prompts"] == 16 and s["stats"]["far"]["BF16C"]["tokens"] > 0
    assert (tmp_path / "res/code_sha.txt").read_text().count("splice_e55.py") == 1


def test_stub_session_finds_planted_failures(tmp_path):
    assert run_stub(tmp_path / "a", "--stub-effect BF16C=0.02")["H16"] == "KILL16"
    assert run_stub(tmp_path / "b", "--stub-effect TIED8=0.003")["H8"] == "KILL8"
    assert run_stub(tmp_path / "c", "--stub-effect E5M2=0")["decision"] == "INVALID"     # V6
    s = run_stub(tmp_path / "d", "--stub-effect DET=0.001")                              # the pre-flight stops it
    assert s["decision"] == "INVALID" and any("DET" in p for p in s["preflight"])
    s = run_stub(tmp_path / "e", "--stub-effect DET=0.00005")                            # not bit-identical, tiny
    assert s["decision"] == "GO16 GO8" and not s["stats"]["det"]["bit_identical"]


def test_preflight_stops_a_broken_engine_before_the_paid_arms(tmp_path):
    """R-77 F3: a kernel that ignores the impl scale (V5) or an engine that ignores the hook (V7, V5) stops the run
    after the pre-flight, before any arm of the 12 documents."""
    for plant, word in (("ignore_scale", "engine-output gaps"), ("ignore_hook", "nearer the unchanged reference")):
        s = run_stub(tmp_path / plant, f"--stub-plant {plant}")
        rec = json.loads((tmp_path / plant / "res/run_main.json").read_text())
        assert s["decision"] == "INVALID" and rec["arms"] == {} and any(word in p for p in s["preflight"]), plant
        assert "PREFLIGHT STOP" in (tmp_path / plant / "res/session.log").read_text()


def test_time_cap_cuts_the_reported_steps_and_keeps_the_decision(tmp_path):
    """The decision arms (about 80 s here) pass the soft limit (120 - 10 - 60 = 50 s), so every reported step is cut,
    and the frozen rule still decides on the decision arms."""
    s = run_stub(tmp_path, "--stub-seconds 1.0", {"POD_LIMIT": "120", "AN_RESERVE": "10"})
    assert {c["step"] for c in s["cut"]} == {"BF16R", "JOINT8", "GEN", "TIED8_UP"}
    assert s["decision"] == "GO16 GO8" and "BF16R" not in s["stats"]["delta"]


def test_a_failed_reported_step_keeps_the_decision(tmp_path):
    """R-77 F2: a reported step that fails does not stop the later steps or change the decision."""
    s = run_stub(tmp_path, "--stub-plant gen_error")
    assert s["decision"] == "GO16 GO8" and [e["step"] for e in s["step_errors"]] == ["GEN"]
    assert "TIED8_UP" in s["stats"]["delta"]


def test_sigterm_after_the_decision_arms_keeps_the_decision(tmp_path):
    """R-77 F2: the session timeout (SIGTERM) during a reported step saves the record with the hook evidence, and the
    decision arms still decide."""
    res = tmp_path / "res"
    p = subprocess.Popen([sys.executable, "-B", str(HERE / "run_e55.py"), "--out", str(res), "--stub",
                          "--stub-seconds", "0.3", "--deadline", str(time.time() + 3_600)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 300:
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
    assert an55.summarize(res, DATA)["decision"] == "GO16 GO8"


def test_rehearsal_runs_every_arm_with_synthetic_scales(tmp_path):
    """R-77 B2: the rehearsal has no E-54 scales; synthetic scales let it run the scaled arms, GEN and the rest, and the
    REHEARSAL line shows the validity problems."""
    s = run_stub(tmp_path, "", {"REH": "1"})
    rec = json.loads((tmp_path / "res/run_main.json").read_text())
    assert rec["rehearsal"] and rec["synthetic_scales"]["5"]["tied"]["vs"] == 0.125
    assert set(rec["arms"]) == set(sp.ARMS) | {"DET", "GEN"}
    assert an55.line(s).startswith("REHEARSAL ")


def test_v2_compares_the_pod_code_hashes_with_the_design(tmp_path):
    data = json.loads(DATA.read_text())
    main = {"stub": False, "gpu": "NVIDIA H100 80GB HBM3", "status": "ok", "docs": [], "doc_lengths": {},
            "tok_check": {"ok": True}, "hooks": {"hooked": []}}
    (tmp_path / "code_manifest.json").write_text(json.dumps({n: "a" * 64 for n in an55.CODE_FILES}))
    (tmp_path / "code_sha.txt").write_text("".join(f"{'a' * 64}  {n}\n" for n in an55.CODE_FILES))
    v, _ = an55.validity(tmp_path, data, main, {}, rehearsal=True)
    assert not any("code hash" in p for p in v["V2"])
    (tmp_path / "code_sha.txt").write_text("".join(f"{'b' * 64}  {n}\n" for n in an55.CODE_FILES))
    v, _ = an55.validity(tmp_path, data, main, {}, rehearsal=True)
    assert sum("code hash" in p for p in v["V2"]) == len(an55.CODE_FILES)


def test_make_manifest_hashes_the_committed_files():
    import make_manifest as mm
    m = mm.manifest("HEAD")
    assert set(m) == set(an55.CODE_FILES)
    assert m["scales_e54.json"] == an55.SCALES_SHA256 and m["data_e55.json"] == an55.DATA_SHA256


def test_manifest_is_accepted_by_the_ledger(tmp_path):
    import ledger
    s = run_stub(tmp_path)
    m = an55.manifest(s, tmp_path / "res", "0" * 40)
    m["hardware"].update(driver="stub", cuda="stub")
    assert ledger.validate(m, set()) == []

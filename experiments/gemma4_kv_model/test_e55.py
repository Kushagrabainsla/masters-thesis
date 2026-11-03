"""E-55 tests (README Section 12). Run: uv run --no-project --with numpy --with torch --with pytest python -B -m pytest
-q -p no:cacheprovider experiments/gemma4_kv_model/test_e55.py"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
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


@pytest.mark.skipif(not (CAP / "p0/L05.pt").exists(), reason="the E-54 captures are not on this machine")
def test_transforms_reproduce_the_e54_errors_on_real_captures():
    """The splice numerics equal the frozen E-54 analysis: BF16C, STOCK8, TIED8, JOINT8 and TIED8_UP give the E-54
    errors of prompt 0 (layers 5 and 41) to 1e-4 relative, and every V5 check passes."""
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


def test_check_transform_catches_planted_faults():
    q, k, v, w = toy_qkv()
    rot, unrot = sp.rot_dims(512)
    sc = {"w": w, "tied": {"vs": 0.125, "ks": sp.f32(w * 0.125)}, "calstd_joint": {"ks": 0.01, "vs": 0.2}}
    for arm in sp.ARMS:
        s = sp.arm_scales(arm, sc)
        q2, k2, v2 = sp.transform(arm, q, k, v, w, s, rot, unrot)
        assert sp.check_transform(arm, q, k, v, q2, k2, v2, w, s, rot, unrot) == [], arm
    s = sp.arm_scales("TIED8", sc)
    q2, k2, v2 = sp.transform("TIED8", q, k, v, w, s, rot, unrot)
    bad_k = k2.clone()
    bad_k[..., unrot[0]] = sp.fp8(k.float()[..., unrot[0]] / s["ks"]).to(k.dtype)     # the standard K code
    assert any("tie" in b for b in sp.check_transform("TIED8", q, k, v, q2, bad_k, v2, w, s, rot, unrot))
    q2, k2, v2 = sp.transform("BF16C", q, k, v, w, s, rot, unrot)
    assert sp.check_transform("BF16C", q, k, v, q, k2, v2, w, s, rot, unrot)          # the query not scaled
    q2, k2, v2 = sp.transform("STOCK8", q, k, v, w, s, rot, unrot)
    assert sp.check_transform("STOCK8", q, k, v, q2, k, v2, w, s, rot, unrot)          # K not cast


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
    """On a toy module the hooks give the emulated output of every arm (the K descale through impl.scale, the V
    descale on the output), count the tokens, pass V5, and leave no trace after detach."""
    q, k, v, w = toy_qkv()
    t, hq, d = q.shape
    hk = k.shape[1]
    sc = {"w": w, "tied": {"vs": 0.125, "ks": sp.f32(w * 0.125)}, "calstd_joint": {"ks": 0.01, "vs": 0.2}}
    mod = ToyAttention(hq, hk, d)
    base = mod(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1))
    splice = sp.Splice({7: w}, {7: sc}, {7: (hq, hk, d)})
    splice.attach(7, mod)
    rot, unrot = sp.rot_dims(d)
    for arm in sp.ARMS:
        splice.set_arm(arm)
        splice.new_doc("X")
        got = mod(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)) if arm != "JOINT8" else \
            mod(query=q.reshape(t, -1), key=k.reshape(t, -1), value=v.reshape(t, -1))   # keyword call
        s = sp.arm_scales(arm, sc)
        q2, k2, v2 = sp.transform(arm, q, k, v, w, s, rot, unrot)
        want = sp.emulate(q2, k2, v2, torch.arange(t), s, arm, attend).to(torch.bfloat16).reshape(t, -1)
        if arm in sp.SCALED:      # the toy rounds the output to BF16 before the V descale, as the engine does
            want = (attend(q2.float(), k2.float() * s["ks"], v2.float(), torch.arange(t)).to(torch.bfloat16).float()
                    * s["vs"]).to(torch.bfloat16).reshape(t, -1)
        assert torch.equal(got, want), arm
        assert splice.counts[(arm, "X", 7)] == t
        if arm == "BF16":
            assert torch.equal(got, base)
    assert all(v == [] for v in splice.checks.values()) and splice.errors == []
    assert set(splice.gaps) == set(sp.ARMS) and all(g[7] < 0.01 for g in splice.gaps.values())
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
    sc = {"w": w, "tied": {"vs": 0.125, "ks": sp.f32(w * 0.125)}, "calstd_joint": {"ks": 0.01, "vs": 0.2}}
    mod = BrokenAttention(hq, k.shape[1], d)
    splice = sp.Splice({7: w}, {7: sc}, {7: (hq, k.shape[1], d)})
    splice.attach(7, mod)
    for arm in ("STOCK8", "TIED8"):
        splice.set_arm(arm)
        splice.new_doc("X")
        mod(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1))
    assert splice.gaps["STOCK8"][7] < 0.01 and splice.gaps["TIED8"][7] > 0.3


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


def stats(bf16c=(0.0, -1e-5, 1e-5), e5m2=0.005, stock=0.002, tied=0.001, d_ci=(-0.0012, -0.0008)):
    return {"delta": {"BF16C": bf16c[0], "E5M2": e5m2, "STOCK8": stock, "TIED8": tied},
            "ci": {"BF16C": (bf16c[1], bf16c[2])}, "m16": min(an55.M16_ABS, an55.M16_REL * e5m2),
            "D": tied - stock, "ci_D": d_ci, "tol8": max(an55.TOL8_ABS, an55.TOL8_REL * stock)}


def test_decision_rule_gives_each_outcome():
    assert an55.decide16(stats())[0] == "GO16"
    assert an55.decide16(stats(bf16c=(0.02, 0.019, 0.021)))[0] == "KILL16"
    assert an55.decide16(stats(bf16c=(0.003, 0.002, 0.004)))[0] == "PIVOT16"
    assert an55.decide16(stats(bf16c=(0.0004, -0.0001, 0.0009)))[0] == "INCONCLUSIVE16"   # margin 0.0005
    assert an55.decide16(stats(bf16c=(0.0004, -0.0001, 0.0009), e5m2=0.02))[0] == "GO16"  # margin 0.001
    assert an55.decide8(stats())[0] == "GO8"
    assert an55.decide8(stats(tied=0.0021, d_ci=(-0.0002, 0.0004)))[0] == "GO8"          # tolerance 0.0005
    assert an55.decide8(stats(tied=0.004, d_ci=(0.0015, 0.0025)))[0] == "KILL8"
    assert an55.decide8(stats(tied=0.0028, d_ci=(0.0006, 0.001)))[0] == "PIVOT8"
    assert an55.decide8(stats(tied=0.0024, d_ci=(-0.0001, 0.0009)))[0] == "INCONCLUSIVE8"


def test_bootstrap_is_stratified_and_fixed_by_the_seed():
    doms = ["books"] * 4 + ["prose"] * 4 + ["code"] * 4
    a, b = an55.draws(doms, boot=500), an55.draws(doms, boot=500)
    assert a.shape == (500, 12) and np.array_equal(a, b)
    for j, dom in enumerate(["books"] * 4 + ["prose"] * 4 + ["code"] * 4):
        assert all(doms[i] == dom for i in a[:, j])


def test_data_file_is_the_frozen_plan():
    data = json.loads(DATA.read_text())
    if an55.DATA_SHA256 != "FROZEN-AT-S1":
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
    text = (HERE / "session_e55.sh").read_text()
    limit = int(re.search(r"POD_LIMIT:-(\d+)", text).group(1))
    assert limit + 150 < 2_820                                 # the session starts about 2 min after the boot watchdog
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
    s = run_stub(tmp_path)
    assert s["decision"] == "GO16 GO8", s.get("validity")
    assert set(s["stats"]["delta"]) == {"BF16C", "E5M2", "STOCK8", "TIED8", "JOINT8", "TIED8_UP", "BF16_B"}
    assert s["stats"]["gen"]["BF16C"]["prompts"] == 16
    assert (tmp_path / "res/code_sha.txt").read_text().count("splice_e55.py") == 1


def test_stub_session_finds_planted_failures(tmp_path):
    assert run_stub(tmp_path / "a", "--stub-effect BF16C=0.02")["H16"] == "KILL16"
    assert run_stub(tmp_path / "b", "--stub-effect TIED8=0.003")["H8"] == "KILL8"
    assert run_stub(tmp_path / "c", "--stub-effect E5M2=0")["decision"] == "INVALID"     # V6
    s = run_stub(tmp_path / "d", "--stub-effect DET=0.001")                              # a rerun far from BF16
    assert s["decision"] == "INVALID" and any("DET" in p for p in s["validity"]["V3"])
    s = run_stub(tmp_path / "e", "--stub-effect DET=0.00005")                            # not bit-identical, tiny
    assert s["decision"] == "GO16 GO8" and not s["stats"]["det"]["bit_identical"]


def test_time_cap_cuts_the_reported_steps_and_keeps_the_decision(tmp_path):
    """The decision arms (about 65 s here) pass the soft limit (120 - 10 - 60 = 50 s), so every reported step and the
    noise phase are cut, and the frozen rule still decides on the decision arms."""
    s = run_stub(tmp_path, "--stub-seconds 1.0", {"POD_LIMIT": "120", "AN_RESERVE": "10"})
    assert {c["step"] for c in s["cut"]} == {"JOINT8", "GEN", "TIED8_UP"}
    assert "noise: cut" in (tmp_path / "res/session.log").read_text()
    assert s["decision"] == "GO16 GO8" and "BF16_B" not in s["stats"]["delta"]


def test_v2_compares_the_pod_code_hashes_with_the_design(tmp_path):
    data = json.loads(DATA.read_text())
    main = {"stub": False, "gpu": "NVIDIA H100 80GB HBM3", "status": "ok", "docs": [], "doc_lengths": {},
            "tok_check": {"ok": True}, "hooks": {"hooked": []}}
    (tmp_path / "code_manifest.json").write_text(json.dumps({n: "a" * 64 for n in an55.CODE_FILES}))
    (tmp_path / "code_sha.txt").write_text("".join(f"{'a' * 64}  {n}\n" for n in an55.CODE_FILES))
    v = an55.validity(tmp_path, data, main, None, {}, rehearsal=True)
    assert not any("code hash" in p for p in v["V2"])
    (tmp_path / "code_sha.txt").write_text("".join(f"{'b' * 64}  {n}\n" for n in an55.CODE_FILES))
    v = an55.validity(tmp_path, data, main, None, {}, rehearsal=True)
    assert sum("code hash" in p for p in v["V2"]) == len(an55.CODE_FILES)


def test_manifest_is_accepted_by_the_ledger(tmp_path):
    import ledger
    s = run_stub(tmp_path)
    m = an55.manifest(s, tmp_path / "res", "0" * 40)
    m["hardware"].update(driver="stub", cuda="stub")
    assert ledger.validate(m, set()) == []

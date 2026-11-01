"""Tests for E-53 with amendment A1 (no GPU, no vLLM): the FP8 cast rule, the causal GQA reference, the views path, the
recovery of w and of its module form, the identity check, the RoPE check of the capture point, the saturation counts
(A1, B3), similar errors of the two storages on well-scaled Gemma 4 K=V data, the seeded prompts and the special ids
(A1, F4), the hooks and the engine output (A1, B2) on a model tree with the vLLM names, the KV-sharing rule and the
shape flags of the rehearsal (A1, F1), the mode names (A1, F7), the session caps and the watchdog order (A1, B1),
every decision path with its action (PASS, FAIL, INCONCLUSIVE, INVALID; A1, B4), one timing check at the real size,
and a stub session from a clean copy of the upload folders.

Run: uv run --no-project --with numpy --with torch --with pytest python -m pytest -q --rootdir <folder>
     <folder>/test_e53.py -p no:cacheprovider
"""

from __future__ import annotations

import enum
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))

import analyze_e53 as an  # noqa: E402
import capture_e53 as cap  # noqa: E402
import compact_attn as ca  # noqa: E402

UPLOAD = ["gemma4_kv_accuracy", "gemma4_kv_kernel", "runpod"]     # README Section 11: the folders the Pod needs
PY = sys.executable


class Args:
    """The stub flags of capture_e53.py with their defaults."""
    stub_q_scale, stub_q_max, stub_big_k, stub_big_k_dim = 2.0, 0.0, 0.0, 0
    stub_break_identity = stub_nan = stub_swap_heads = False
    heads, kv_heads = 32, 4

    def __init__(self, **kw):
        self.__dict__.update(kw)


def gemma_case(length: int = 512, q_rows: int = 32, seed: int = 1, **kw):
    """One global layer built as in Gemma 4 and rounded as the engine rounds it (capture_e53.stub_layer)."""
    gen = torch.Generator().manual_seed(seed)
    w = (1 + 0.1 * torch.randn(512, generator=gen, dtype=torch.float64)).to(torch.bfloat16).double()
    return w, cap.stub_layer(length, q_rows, w, gen, Args(**kw))


def test_fp8_cast_is_the_vllm_rule() -> None:
    """__NV_SATFINITE and round to nearest even: +-448 at most, 2^-10 is a tie that goes to 0, 2^-9 is the smallest
    subnormal."""
    x = torch.tensor([500.0, -470.0, 448.0, 449.0, 2.0 ** -10, 1.01 * 2.0 ** -10, 2.0 ** -7, 0.3, 0.0])
    y = an.fp8(x)
    assert y.tolist()[:4] == [448.0, -448.0, 448.0, 448.0]
    assert y[4] == 0 and y[5] == 2.0 ** -9 and y[6] == 2.0 ** -7 and abs(float(y[7]) - 0.3) <= 0.3 / 16
    s = an.stats(x, y)
    assert (s["sat"], s["sub"], s["flush"], s["nonfinite"], s["n"]) == (3, 2, 1, 0, 9)
    assert s["max_pre"] == 500.0 and s["max_post"] == 448.0
    assert an.e5m2(torch.tensor([1e6, 0.3])).tolist() == [57344.0, 0.3125]      # (A1, O2) 2 mantissa bits


def test_reference_is_causal_gqa() -> None:
    """attend() at the last position equals the dense GQA reference of compact_attn; at an earlier position it equals
    the reference over the prefix only (the causal mask). (A1) It reads the GQA ratio from the shapes (8 to 1 too)."""
    gen = torch.Generator().manual_seed(2)
    k, v = ca.make_kv(300, 4, torch.ones(512, dtype=torch.float64), gen, torch.float64)
    q = torch.randn(2, 32, 512, generator=gen, dtype=torch.float64) / 512 ** 0.5
    got = an.attend(q, k, v, torch.tensor([299, 150]))
    assert torch.allclose(got[0].double(), ca.gqa_attend(q[0], k, v), atol=1e-6)
    assert torch.allclose(got[1].double(), ca.gqa_attend(q[1], k[:151], v[:151]), atol=1e-6)
    got1 = an.attend(q[:, :8], k[:, :1], v[:, :1], torch.tensor([299, 150]))     # the E2B shape: 8 heads, 1 KV head
    assert torch.allclose(got1[0].double(), ca.gqa_attend(q[0, :8], k[:, :1], v[:, :1]), atol=1e-6)


def test_views_path_is_exact_without_rounding() -> None:
    """The views row read with the views query, then `unpermute`, gives the reference attention (no FP8, float64
    inputs): the compact path itself adds no error."""
    w, c = gemma_case(256, 8)
    k, v, q, qpos = c["k"].double(), c["v"].double(), c["q"].double(), c["qpos"]
    kview, vview = ca.views_of(ca.pack_views(k, v))
    ref = an.attend(q, k, v, qpos)
    got = ca.unpermute(an.attend(ca.views_query(q, w), kview, vview, qpos))
    assert an.rel(got, ref) < 5e-3                       # K_unrot is BF16(w * V), so equal to the BF16 level only
    k_exact = ca.rope(w * v, torch.arange(256).double()[:, None].expand(256, 4))
    ref_exact = an.attend(q, k_exact, v, qpos)
    assert an.rel(ca.unpermute(an.attend(ca.views_query(q, w), *ca.views_of(ca.pack_views(k_exact, v)), qpos)),
                  ref_exact) < 1e-6


def test_w_is_recovered_and_the_data_decide_its_form() -> None:
    w, c = gemma_case()
    k, v = c["k"].float(), c["v"].float()
    w_data = an.w_from_data(k, v)
    un = torch.tensor(ca.UNROT)
    assert an.rel(w_data, w[un].float()) < 1e-2
    form, w_eff, dist = an.w_choice(w_data, w.float())
    assert form == "weight" and dist["weight"] < an.W_MATCH < dist["one_plus_weight"]
    assert torch.equal(w_eff, w.float())
    form, w_eff, _ = an.w_choice(w_data, (w - 1).float())       # a module that stores w - 1 (GemmaRMSNorm style)
    assert form == "one_plus_weight" and an.rel(w_eff, w.float()) < 1e-6
    form, w_eff, dist = an.w_choice(w_data, None)
    assert form == "data" and dist == {} and an.rel(w_eff[un], w[un].float()) < 1e-2


def test_identity_check_passes_on_k_eq_v_data_and_fails_on_broken_data() -> None:
    w, c = gemma_case()
    assert an.identity_err(c["k"].float(), c["v"].float(), w.float()) < 3e-3       # the BF16 level
    assert an.identity_token_max(c["k"].float(), c["v"].float(), w.float()) < 1e-2   # (A1, F5) per token
    _, broken = gemma_case(stub_break_identity=True)
    assert an.identity_err(broken["k"].float(), broken["v"].float(), w.float()) > an.ID_TOL
    assert an.identity_err(c["k"].float(), c["v"].float(), (w + 1).float()) > an.ID_TOL   # the wrong form of w
    k1 = c["k"].float().clone()
    k1[0, 0, ca.UNROT] *= 3                                                      # a break at one token only
    assert an.identity_err(k1, c["v"].float(), w.float()) < 0.2 < an.identity_token_max(k1, c["v"].float(), w.float())


def test_rope_check_finds_the_capture_point() -> None:
    """After RoPE the rotated pairs fit the Gemma 4 rotation; a capture before RoPE (K = w * V) does not."""
    w, c = gemma_case()
    k, v = c["k"].float(), c["v"].float()
    r = an.rope_check(k, v, c["kpos"], w.float())
    assert r["after_rope"] and r["resid"] < 5e-3 and r["resid_no_rope"] > 0.5 and r["w_rot_vs_weight"] < 1e-2
    pre = an.rope_check(an.bf16(w.float() * v), v, c["kpos"], w.float())
    assert not pre["after_rope"] and pre["resid"] > 0.3


def test_errors_are_similar_on_well_scaled_data() -> None:
    w, c = gemma_case(640, 32)
    m = an.layer_metrics(c, w.float(), "cpu")
    assert 0.01 < m["err_std"] < 0.1 and 0.8 < m["err_cmp"] / m["err_std"] < 1.25, (m["err_std"], m["err_cmp"])
    assert m["w_form"] == "weight" and m["ref_finite"] and m["identity_err"] < an.ID_TOL and m["rope"]["after_rope"]
    assert m["store_std"]["n"] == 640 * 4 * 1024 and m["store_cmp"]["n"] == 640 * 4 * 640
    assert m["store_std"]["sat"] == m["store_cmp"]["sat"] == 0 and m["qv_max"] < 10
    assert m["kview_max_post"] <= 448 and m["vview_max_post"] <= 448 and m["nonfinite_fp16"] == 0
    assert m["engine_gap"] < 0.01 and m["w_module"] and m["w_match"]                 # (A1, B2, F5)
    assert 0.8 < m["err_cmp_fp16"] / m["err_std_fp16"] < 1.25
    # (A1, F6) both storages hold the same fp8(V): the value-path errors are equal
    assert m["err_std_value"] == pytest.approx(m["err_cmp_value"], rel=0.05) and m["err_std_score"] > 0
    assert m["err_cmp_e5m2"] / m["err_std"] > 1.5                                    # (A1, O2) the harness sees 2 x


def test_saturation_is_counted() -> None:
    """(A1, B3) K values of 600 on one rotated dim for 64 tokens: both storages saturate the same 256 values, so the
    counts are equal and the count rule passes (the old share rule failed it: 640 against 1,024 values per token). On
    one unrotated dim only the standard storage saturates (the compact rows hold V there)."""
    w, c = gemma_case(640, 16, stub_big_k=600.0)
    m = an.layer_metrics(c, w.float(), "cpu")
    assert m["store_std"]["sat"] == m["store_cmp"]["sat"] == 64 * 4 and m["store_std_k"]["sat"] == 256
    assert m["store_std_krot"]["sat"] == m["store_cmp_krot"]["sat"] == 256 and m["store_std_kunrot"]["sat"] == 0
    assert m["store_std"]["max_pre"] == 600.0 and m["store_std"]["max_post"] == 448.0
    s = an.layer_summary([m])
    assert s["sat_cmp_count"] == s["sat_std_count"] == 256 and s["pass"]["sat"]
    assert s["sat_share_krot_cmp"] == s["sat_share_krot_std"] and math.isclose(s["sat_cmp"] / s["sat_std"], 1.6)
    w, c = gemma_case(640, 16, stub_big_k=600.0, stub_big_k_dim=ca.UNROT[0])
    s = an.layer_summary([an.layer_metrics(c, w.float(), "cpu")])
    assert s["sat_std_count"] == s["sat_std_kunrot_count"] == 256 and s["sat_cmp_count"] == 0 and s["pass"]["sat"]


def test_prompts_are_seeded_hashed_and_free_of_special_ids(tmp_path: Path) -> None:
    root = tmp_path / "site"
    for i in range(12):
        (root / f"pkg{i % 3}").mkdir(parents=True, exist_ok=True)
        (root / f"pkg{i % 3}" / f"m{i}.py").write_text(f"# module {i}\n" + "def f(x):\n    return x + 1\n" * 150)
        (root / f"pkg{i % 3}" / f"README{i}.md").write_text(f"# Doc {i}\n" + "Some text here.\n" * 200)
    (root / "pkg0" / "LICENSE.md").write_text("x" * 5000)
    (root / "pkg1" / "tok.md").write_text("The text <bos> is a special-token string.\n" * 100)   # (A1, F4)
    src = cap.sources({"vllm_py": [root / "pkg0"], "torch_py": [root / "pkg1"], "docs": [root]})
    assert all("LICENSE" not in f.name for f in src["docs"]) and len(src["mixed"]) == 4 + 4 + 13
    kw = dict(n=4, tokens=4096, special=cap.TOY_SPECIAL)
    a = cap.build_prompts(src, cap.toy_encode, 2, root, seed=53, **kw)
    b = cap.build_prompts(src, cap.toy_encode, 2, root, seed=53, **kw)
    c = cap.build_prompts(src, cap.toy_encode, 2, root, seed=54, **kw)
    assert [p["token_sha256"] for p in a] == [p["token_sha256"] for p in b] != [p["token_sha256"] for p in c]
    assert all(p["n_tokens"] == len(p["token_ids"]) == 4096 and p["token_ids"][0] == 2 for p in a)
    assert [p["family"] for p in a] == list(cap.FAMILIES) and all(len(p["files"]) >= 2 for p in a)
    assert all(f["tokens"] <= cap.FILE_TOKENS + 2 and len(f["sha256"]) == 64 for p in a for f in p["files"])
    assert a[0]["token_sha256"] == cap.ids_sha(a[0]["token_ids"])
    assert all(p["special_after_0"] == 0 and "tok.md" not in str(p["files"]) for p in a)
    # (A1, F4) a docs family of one README (3,002 tokens) and the special-token file: prompt 2 must visit both
    src2 = dict(src, docs=[root / "pkg1" / "tok.md", root / "pkg0" / "README0.md"])
    p2 = cap.build_prompts(src2, cap.toy_encode, 2, root, seed=53, **kw)[2]
    assert p2["skipped_special"] == 1 and p2["special_after_0"] == 0 and p2["n_tokens"] == 4096
    raw = cap.build_prompts(src2, cap.toy_encode, 2, root, seed=53, n=4, tokens=4096)[2]    # without the filter
    assert raw["token_ids"][1:].count(2) > 0 and raw["skipped_special"] == 0     # the second BOS id gets in


def test_global_layers_rule_and_mode_names() -> None:
    """(A1, F1) The global layers that own a KV cache: the E2B config (HF, revision 3e22461f: 35 layers, a global
    layer every 5th from 4, 20 KV-shared layers) gives 4, 9 and 14; the 31B config gives its 10 global layers.
    (A1, F7) An IntEnum mode gives its name, not "0"."""
    e2b = ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(35)]
    assert cap.global_layers(e2b, 35, 20) == [4, 9, 14]
    m31 = ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)]
    assert cap.global_layers(m31, 60, 0) == [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]

    class Mode(enum.IntEnum):
        NONE = 0

    class Graph(enum.Enum):
        NONE = 0

    assert str(Mode.NONE) == "0" and cap.mode_name(Mode.NONE) == "NONE" and cap.mode_name(Graph.NONE) == "NONE"
    assert cap.mode_name("auto") == "auto"


class Attention(nn.Module):
    """A stand-in with the class name and attributes of the vLLM Attention module (attention.py 333 to 339). Its
    output is its query, so the output hook must record the q rows."""

    def __init__(self, layer: int, sliding: bool):
        super().__init__()
        self.layer_name = f"language_model.model.layers.{layer}.self_attn.attn"
        self.num_heads, self.num_kv_heads, self.head_size = 32, (16 if sliding else 4), (256 if sliding else 512)
        self.head_size_v = self.head_size
        self.sliding_window = 1024 if sliding else None

    def forward(self, query, key, value):
        return query


class Norm(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight, self.has_weight, self.variance_epsilon = nn.Parameter(w), True, 1e-6


class Gemma4Attention(nn.Module):
    """A stand-in of Gemma4Attention: forward(positions, hidden_states) calls self.attn(q, k, v) (gemma4.py 572)."""

    def __init__(self, layer: int, sliding: bool):
        super().__init__()
        self.is_sliding, self.is_kv_shared_layer = sliding, False
        self.attn, self.k_norm = Attention(layer, sliding), Norm(torch.full((512,), 1.0 + layer))

    def forward(self, positions, hidden_states):
        return self.attn(*hidden_states)


class Layer(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.self_attn = Gemma4Attention(layer, layer not in (5, 11))


def fake_model() -> nn.Module:
    root, lm, inner = nn.Module(), nn.Module(), nn.Module()
    inner.layers = nn.ModuleList([Layer(i) for i in range(12)])
    lm.model = inner
    root.language_model = lm
    return root


def test_hooks_record_the_global_layers_and_the_engine_output(tmp_path: Path) -> None:
    """register() hooks layers 5 and 11 only; a prefill in two chunks gives the last q_rows query rows, (A1, B2) the
    output rows of the same positions, and every key row in position order; the k_norm weights are read; the files
    have a SHA-256."""
    model, store = fake_model(), cap.Store(q_rows=4)
    reg = cap.register(model, store)
    assert [h["layer"] for h in reg["hooked"]] == [5, 11] and sorted(reg["k_norm"]) == [5, 11]
    assert float(reg["k_norm"][11][0]) == 12.0 and not cap.hook_problems(reg["hooked"], [5, 11], (32, 4, 512), 2)
    gen = torch.Generator().manual_seed(5)
    q, k, v = (torch.randn(10, h * 512, generator=gen).to(torch.bfloat16) for h in (32, 4, 4))
    store.begin(10)
    for lo, hi in ((0, 6), (6, 10)):
        for i in (5, 11):
            model.language_model.model.layers[i].self_attn(positions=torch.arange(lo, hi),
                                                           hidden_states=(q[lo:hi], k[lo:hi], v[lo:hi]))
    data = store.end()
    assert cap.check_prompt(data, [5, 11], 10, 4, store.errors) == []
    assert torch.equal(data[5]["q"], q[6:].reshape(4, 32, 512)) and torch.equal(data[11]["v"], v.reshape(10, 4, 512))
    assert torch.equal(data[5]["o"], data[5]["q"]) and data[5]["src"] == ["positions"]
    files = cap.save_prompt(tmp_path, 0, data)
    assert set(files) == {"5", "11"} and all(len(f["sha256"]) == 64 for f in files.values())
    back = torch.load(tmp_path / "p0" / "L05.pt", weights_only=True)
    assert torch.equal(back["k"], data[5]["k"]) and torch.equal(back["o"], data[5]["o"]) and back["layer"] == 5
    store.begin(10)                                      # without the parent hook: the row order, and a bad shape
    model.language_model.model.layers[5].self_attn.attn(q[:8], k[:8], v[:8])
    bad = cap.check_prompt(store.end(), [5, 11], 10, 4, [])
    assert any("layer 11: no tensors" in b for b in bad) and any("k shape" in b for b in bad)
    assert cap.check_prompt({}, [5, 11], 10, 4, []) == ["no global-layer tensors were recorded"]
    no_o = {5: {key: val for key, val in data[5].items() if key != "o"}}
    assert any("no o rows" in b for b in cap.check_prompt(no_o, [5], 10, 4, []))
    wrong = [dict(h, num_kv_heads=8) for h in reg["hooked"]]
    assert cap.hook_problems(wrong, [5, 11, 17])[0].startswith("hooked layers")
    assert cap.hook_problems(reg["hooked"], [5, 11], (32, 4, 512), 10)[0].startswith("the config has 2")


def run_stub(res: Path, *extra: str, check: bool = True) -> dict:
    r = subprocess.run([PY, str(HERE / "capture_e53.py"), "--stub", "--out", str(res), "--capture", str(res / "cap"),
                        *extra], capture_output=True, text=True, timeout=300)
    if check:
        assert r.returncode == 0, r.stderr[-2000:]
    an.compute_metrics(res / "cap", res, "cpu")
    return analyze(res)


def analyze(res: Path) -> dict:
    s = an.summarize(res)
    s["decision"] = an.decide(s)
    return s


def edit_json(path: Path, fn) -> None:
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def edit_metrics(res: Path, fn) -> dict:
    edit_json(res / "metrics.json", fn)
    return analyze(res)


def test_pass(tmp_path: Path) -> None:
    s = run_stub(tmp_path)
    assert s["decision"].startswith("PASS (2 global layers"), s["decision"]
    assert s["decision"].endswith("action: week 2 of the build starts (D-88))")
    assert all(s["checks"][k] for k in ("V1", "V2", "V3", "V4", "V5")) and s["checks"]["_identity_max"] < 3e-3
    assert s["checks"]["_engine_gap_max"] < 0.01 and s["checks"]["_sha_checked"] == 4
    for v in s["layers"].values():
        assert 0.8 < v["med_ratio_used"] < 1.25 and v["w_forms"] == ["weight"] and v["rope_after"] and v["prompts"] == 2
        assert len(v["ratio"]) == len(v["ratio_fp16"]) == 2 and v["exploratory"]["ratio_e5m2"] > 1.5
    assert len(s["ledger_metrics"]["ratio"]["values"]) == 4 and s["prompts"][0]["token_sha256"]


def test_module_forms_of_w(tmp_path: Path) -> None:
    """The data decide the form: a module that stores w - 1 gives one_plus_weight. (A1, F5) Without a module weight V3
    fails, because a fit of w to the same data cannot show a wrong module weight."""
    s = run_stub(tmp_path / "one", "--stub-w-form", "one_plus_weight")
    assert s["decision"].startswith("PASS") and all(v["w_forms"] == ["one_plus_weight"] for v in s["layers"].values())
    s = run_stub(tmp_path / "none", "--stub-w-form", "none")
    assert s["decision"].startswith("INVALID (V3") and "action: report the recovered w" in s["decision"], s["decision"]


def test_fail_path(tmp_path: Path) -> None:
    run_stub(tmp_path)

    def worse(d):                                                        # layer 11: 2.5 x the standard error
        for m in d["prompts"].values():
            m["11"]["err_cmp"] = 2.5 * m["11"]["err_std"]

    s = edit_metrics(tmp_path, worse)
    assert s["decision"].startswith("FAIL (layer 11: median per-prompt ratio 2.50 above 2.0; action: stop the build")


def test_inconclusive_paths(tmp_path: Path) -> None:
    """(A1) A |q'| above the FP16 range or an FP16 value that is not finite: INCONCLUSIVE with the action "pre-scale
    the query" (R-73 B4). The per-prompt ratios (F3) and the FP16 ratio (F3) enter the PASS tests. The count rule
    (B3) fails only if the compact rows saturate more values."""
    s = run_stub(tmp_path / "q", "--stub-q-max", "50000")               # |q'| = 50,000 x w, above 32,752
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: largest |q'|") and "(the FP16 range)" in s["decision"]
    assert not s["decision"].startswith("FAIL") and s["q_max"] > an.Q_RANGE, s["decision"]
    res = tmp_path / "r"
    run_stub(res)
    s = edit_metrics(res, lambda d: d["prompts"]["1"]["5"].update(qv_max=40000.0))    # the range alone
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: largest |q'| 40,000.0 above 32,752.0 (the FP16 range); "
                                    "action: pre-scale the query"), s["decision"]
    run_stub(res)

    def set_errors(layer: str, f32: tuple, f16: tuple):
        def fn(d):
            for p, (a, b) in enumerate(zip(f32, f16)):
                m = d["prompts"][str(p)][layer]
                m["err_cmp"], m["err_cmp_fp16"] = a * m["err_std"], b * m["err_std_fp16"]
        return fn

    s = edit_metrics(res, set_errors("5", (1.5, 1.5), (1.0, 1.0)))
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: median per-prompt ratio 1.500 above 1.25"), s["decision"]
    assert "action: the user decides" in s["decision"]
    run_stub(res)
    s = edit_metrics(res, set_errors("5", (1.0, 1.0), (1.6, 1.6)))     # only the FP16 emulation is worse
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: median per-prompt ratio 1.600"), s["decision"]
    run_stub(res)
    s = edit_metrics(res, set_errors("5", (0.3, 2.1), (0.3, 2.1)))     # the median holds, the largest does not
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: largest per-prompt ratio 2.100 above 2.0"), s["decision"]
    run_stub(res)

    def r73_case(d):            # R-73 F3: the ratio of the maxima (1.5) misses a prompt with 3 x its own error
        a, b = d["prompts"]["0"]["5"], d["prompts"]["1"]["5"]
        a.update(err_std=0.05, err_cmp=0.15, err_std_fp16=0.05, err_cmp_fp16=0.15)
        b.update(err_std=0.10, err_cmp=0.10, err_std_fp16=0.10, err_cmp_fp16=0.10)

    s = edit_metrics(res, r73_case)
    assert s["layers"]["5"]["ratio_of_maxima"] == pytest.approx(1.5) and not s["decision"].startswith("FAIL")
    assert "layer 5: largest per-prompt ratio 3.000 above 2.0" in s["decision"], s["decision"]
    run_stub(res)

    def sat(d):                                                         # more saturated compact values
        d["prompts"]["0"]["5"]["store_cmp"]["sat"] = 7

    s = edit_metrics(res, sat)
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: 7 saturated compact values against 0 standard")
    run_stub(res)
    s = edit_metrics(res, lambda d: d["prompts"]["0"]["11"].update(qv_max=10000.0))
    assert s["decision"].startswith("INCONCLUSIVE (layer 11: largest |q'| 10,000.0 above 6,550.0 (the 10 times")
    assert "action: pre-scale the query" in s["decision"]
    run_stub(res)
    s = edit_metrics(res, lambda d: d["prompts"]["1"]["5"].update(nonfinite_fp16=3))
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: 3 FP16 values not finite; action: pre-scale the query")


def test_invalid_paths(tmp_path: Path) -> None:
    s = run_stub(tmp_path / "gpu", "--stub-gpu", "STUB NVIDIA L4")
    assert s["decision"].startswith("INVALID (V1"), s["decision"]
    s = run_stub(tmp_path / "ff", "--stub-no-tensors", check=False)
    assert s["decision"].startswith("INVALID (V2: fail fast after the first prompt: no global-layer tensors")
    assert "action: fix the cause, then one rerun under a new S4" in s["decision"]
    s = run_stub(tmp_path / "err", "--stub-fail", check=False)
    assert s["decision"].startswith("INVALID (V2") and "stub failure" in s["decision"], s["decision"]
    s = run_stub(tmp_path / "id", "--stub-break-identity")
    assert s["decision"].startswith("INVALID (V3"), s["decision"]
    s = run_stub(tmp_path / "nan", "--stub-nan")
    assert s["decision"].startswith("INVALID (V4"), s["decision"]
    s = run_stub(tmp_path / "swap", "--stub-swap-heads")                # (A1, B2) a wrong head mapping
    assert s["decision"].startswith("INVALID (V5: V5 engine gap above 0.05"), s["decision"]
    assert "action: the reference does not model the engine" in s["decision"] and s["checks"]["_engine_gap_max"] > 0.2
    run_stub(tmp_path / "plan")                                         # a real run must use the frozen plan
    edit_json(tmp_path / "plan" / "capture_run.json", lambda d: d.update(stub=False))
    s = analyze(tmp_path / "plan")
    assert s["decision"].startswith("INVALID (V2") and "frozen plan" in s["decision"], s["decision"]
    assert "vLLM 'STUB'" in " ".join(s["checks"]["_v2_problems"])        # (A1, F5) the engine checks of a real run
    run_stub(tmp_path / "miss")
    (tmp_path / "miss" / "cap" / "p1" / "L11.pt").unlink()
    an.compute_metrics(tmp_path / "miss" / "cap", tmp_path / "miss", "cpu")
    s = analyze(tmp_path / "miss")
    assert s["decision"].startswith("INVALID (V2") and "prompt 1 layer 11: no metrics" in s["decision"]
    run_stub(tmp_path / "sha")                                          # (A1, F2) a capture file that changed
    edit_json(tmp_path / "sha" / "capture_run.json", lambda d: d["prompts"][1]["files"]["5"].update(sha256="0" * 64))
    an.compute_metrics(tmp_path / "sha" / "cap", tmp_path / "sha", "cpu")
    s = analyze(tmp_path / "sha")
    assert s["decision"].startswith("INVALID (V2") and "another SHA-256: ['p1 L5']" in s["decision"], s["decision"]
    run_stub(tmp_path / "spec")                                         # (A1, F4) a special id inside a prompt
    edit_json(tmp_path / "spec" / "prompts.json", lambda d: d["prompts"][0].update(special_after_0=1))
    s = analyze(tmp_path / "spec")
    assert "prompt 0 has 1 special ids after position 0" in s["decision"], s["decision"]
    assert analyze(tmp_path / "empty")["decision"].startswith("INVALID (V1, V2, V3, V4, V5")


def test_rehearsal_shapes_and_engine_checks(tmp_path: Path) -> None:
    """(A1, R-73 F1) The rehearsal shapes (8 query heads, 1 KV head, 3 global layers, as the E2B global layers that own
    a KV cache) run through the same capture and analysis. A rehearsal record skips the frozen-plan check but keeps the
    engine checks; its decision starts with REHEARSAL."""
    res = tmp_path / "e2b"
    s = run_stub(res, "--heads", "8", "--kv-heads", "1", "--stub-layers", "3")
    assert s["decision"].startswith("PASS (3 global layers"), s["decision"]
    assert s["plan"]["heads"] == 8 and s["plan"]["global_layers"] == 3 and s["checks"]["_engine_gap_max"] < 0.01
    eng = {"compilation_mode": "NONE", "cudagraph_mode": "NONE", "cache_dtype": "auto"}
    edit_json(res / "capture_run.json", lambda d: d.update(stub=False, rehearsal=True, vllm=an.VLLM_VERSION, engine=eng))
    s = analyze(res)
    assert s["decision"].startswith("REHEARSAL PASS (3 global layers"), s["decision"]
    edit_json(res / "capture_run.json", lambda d: d["engine"].update(compilation_mode="0"))
    s = analyze(res)
    assert s["decision"].startswith("REHEARSAL INVALID (V2") and "compilation_mode '0'" in s["decision"]
    s = run_stub(tmp_path / "swap1", "--heads", "8", "--kv-heads", "1", "--stub-layers", "3", "--stub-swap-heads")
    assert s["decision"].startswith("INVALID (V5"), s["decision"]       # one KV head: the query heads are rolled


def session_defaults() -> dict:
    text = (HERE / "session_e53.sh").read_text()
    return {k: int(v) for k, v in re.findall(r"^(\w+)=\$\{\w+:-(\d+)\}", text, re.M)}


def test_session_caps_keep_the_watchdog_order() -> None:
    """(A1, R-73 B1) The step caps fit inside the session watchdog; the session watchdog fires before the boot
    watchdog (the session starts about 140 s after boot, E-52); the worst case is at most the $3 cap (a 15 min pull
    plus the boot watchdog, at $3.99/h), and the S5 rehearsal stays inside its $0.60 cap (RTX 4090 at $0.89/h)."""
    d = session_defaults()
    assert (d["DL_TIMEOUT"], d["CAP_TIMEOUT"], d["AN_TIMEOUT"], d["POD_LIMIT"], d["AUTO_STOP_DELAY"]) == \
        (420, 600, 240, 1440, 300)
    boot, start, setup, gpu_wait = 1740, 140, 60, 60
    assert d["DL_TIMEOUT"] + d["CAP_TIMEOUT"] + d["AN_TIMEOUT"] + setup + gpu_wait <= d["POD_LIMIT"]
    assert start + d["POD_LIMIT"] < boot
    assert (15 * 60 + boot) / 3600 * 3.99 <= 3.00                        # 44 min, $2.93
    assert (15 * 60 + 1500) / 3600 * 0.89 <= 0.60                        # the rehearsal: 40 min, $0.59
    assert start + 1200 < 1500                                          # its POD_LIMIT 1,200 s before its CAP 1,500 s


def test_timing_at_the_real_size() -> None:
    """S3 (E-38): one prompt and layer at the real size (8,192 tokens, 256 queries) on the Mac CPU, with the A1
    metrics (11 attention passes), and the hook copy of one real-size layer. The H100 runs 40 of them."""
    w, c = gemma_case(8192, 256, seed=7)
    t = time.time()
    m = an.layer_metrics(c, w.float(), "cpu")
    dt = time.time() - t
    assert dt < 300 and 0.8 < m["err_cmp"] / m["err_std"] < 1.25 and m["identity_err"] < 3e-3, dt
    assert m["engine_gap"] < 0.01
    store = cap.Store(q_rows=256)
    q = torch.zeros(8192, 32 * 512, dtype=torch.bfloat16)
    k = torch.zeros(8192, 4 * 512, dtype=torch.bfloat16)
    store.begin(8192)
    t = time.time()
    store.attn_hook(5, 32, 4, 512)(None, (q, k, k), {})
    store.out_hook(5, 32, 512)(None, (q, k, k), q)
    data = store.end()
    assert time.time() - t < 30 and list(data[5]["q"].shape) == list(data[5]["o"].shape) == [256, 32, 512]
    print(f"layer_metrics at the real size: {dt:.1f} s on the CPU")


def clean_copy(tmp_path: Path) -> Path:
    code = tmp_path / "code"
    for sub in UPLOAD:
        shutil.copytree(HERE.parent / sub, code / sub,
                        ignore=shutil.ignore_patterns("results", "dryrun", "__pycache__", "*.gz", ".pytest_cache"))
    return code


def session(code: Path, res: Path, **extra: str) -> subprocess.CompletedProcess:
    e = dict(os.environ, STUB="1", RES=str(res), CAPDIR=str(res.parent / (res.name + "_cap")), CODE=str(code),
             AUTO_STOP="0", PATH=f"{Path(PY).parent}:{os.environ.get('PATH', '')}", **extra)
    return subprocess.run([shutil.which("sh"), str(code / "gemma4_kv_accuracy" / "session_e53.sh")], env=e,
                          capture_output=True, text=True, timeout=600)


def test_stub_session(tmp_path: Path) -> None:
    code, res = clean_copy(tmp_path), tmp_path / "e53"
    r = session(code, res)
    assert r.returncode == 0, r.stderr[-2000:]
    log = (res / "session.log").read_text()
    assert "capture: exit 0" in log and "analysis: exit 0" in log and "decision: PASS (2 global layers" in log, log
    assert (res / "summary.json").exists() and (res / "prompts.json").exists() and not list(res.glob("**/*.pt"))
    shas = (res / "capture_sha.txt").read_text().split("\n")              # (A1, F2) the copy check of CAPDIR
    assert sorted(line.split()[1] for line in shas if line) == ["k_norm.pt", "p0/L05.pt", "p0/L11.pt", "p1/L05.pt",
                                                               "p1/L11.pt"]
    assert "captures: 5 files" in log
    session(code, res)
    assert "capture: ok before, skip" in (res / "session.log").read_text()
    session(code, tmp_path / "failed", STUB_ARGS="--stub-fail")
    assert "decision: INVALID (V2" in (tmp_path / "failed" / "session.log").read_text()
    session(code, tmp_path / "ff", STUB_ARGS="--stub-no-tensors")
    log = (tmp_path / "ff" / "session.log").read_text()
    assert "capture: exit 3" in log and "decision: INVALID (V2: fail fast" in log, log


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

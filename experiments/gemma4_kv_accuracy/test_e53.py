"""Tests for E-53 (no GPU, no vLLM): the FP8 cast rule, the causal GQA reference, the views path, the recovery of w
and of its module form, the identity check, the RoPE check of the capture point, the saturation counts, similar
errors of the two storages on well-scaled Gemma 4 K=V data, the seeded prompts, the hooks on a model tree with the
vLLM names, every decision path (PASS, FAIL, INCONCLUSIVE, INVALID) on stub inputs, one timing check at the real size,
and a stub session from a clean copy of the upload folders.

Run: uv run --no-project --with numpy --with torch --with pytest python -m pytest -q --rootdir <folder>
     <folder>/test_e53.py -p no:cacheprovider
"""

from __future__ import annotations

import json
import math
import os
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
    stub_q_scale, stub_q_max, stub_big_k, stub_break_identity, stub_nan = 2.0, 0.0, 0.0, False, False

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


def test_reference_is_causal_gqa() -> None:
    """attend() at the last position equals the dense GQA reference of compact_attn; at an earlier position it equals
    the reference over the prefix only (the causal mask)."""
    gen = torch.Generator().manual_seed(2)
    k, v = ca.make_kv(300, 4, torch.ones(512, dtype=torch.float64), gen, torch.float64)
    q = torch.randn(2, 32, 512, generator=gen, dtype=torch.float64) / 512 ** 0.5
    got = an.attend(q, k, v, torch.tensor([299, 150]))
    assert torch.allclose(got[0].double(), ca.gqa_attend(q[0], k, v), atol=1e-6)
    assert torch.allclose(got[1].double(), ca.gqa_attend(q[1], k[:151], v[:151]), atol=1e-6)


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
    _, broken = gemma_case(stub_break_identity=True)
    assert an.identity_err(broken["k"].float(), broken["v"].float(), w.float()) > an.ID_TOL
    assert an.identity_err(c["k"].float(), c["v"].float(), (w + 1).float()) > an.ID_TOL   # the wrong form of w


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


def test_saturation_is_counted() -> None:
    """K values of 600 on one rotated dim for 64 tokens: both storages saturate those 256 values; the compact rows have
    fewer values, so their share is 1,024 / 640 = 1.6 times larger."""
    w, c = gemma_case(640, 16, stub_big_k=600.0)
    m = an.layer_metrics(c, w.float(), "cpu")
    assert m["store_std"]["sat"] == m["store_cmp"]["sat"] == 64 * 4 and m["store_std_k"]["sat"] == 256
    assert m["store_std"]["max_pre"] == 600.0 and m["store_std"]["max_post"] == 448.0
    s = an.layer_summary([m])
    assert math.isclose(s["sat_cmp"] / s["sat_std"], 1.6, rel_tol=1e-9) and s["pass"]["sat"]


def test_prompts_are_seeded_and_hashed(tmp_path: Path) -> None:
    root = tmp_path / "site"
    for i in range(12):
        (root / f"pkg{i % 3}").mkdir(parents=True, exist_ok=True)
        (root / f"pkg{i % 3}" / f"m{i}.py").write_text(f"# module {i}\n" + "def f(x):\n    return x + 1\n" * 150)
        (root / f"pkg{i % 3}" / f"README{i}.md").write_text(f"# Doc {i}\n" + "Some text here.\n" * 200)
    (root / "pkg0" / "LICENSE.md").write_text("x" * 5000)
    src = cap.sources({"vllm_py": [root / "pkg0"], "torch_py": [root / "pkg1"], "docs": [root]})
    assert all("LICENSE" not in f.name for f in src["docs"]) and len(src["mixed"]) == 4 + 4 + 12
    a = cap.build_prompts(src, cap.toy_encode, 2, root, seed=53, n=4, tokens=4096)
    b = cap.build_prompts(src, cap.toy_encode, 2, root, seed=53, n=4, tokens=4096)
    c = cap.build_prompts(src, cap.toy_encode, 2, root, seed=54, n=4, tokens=4096)
    assert [p["token_sha256"] for p in a] == [p["token_sha256"] for p in b] != [p["token_sha256"] for p in c]
    assert all(p["n_tokens"] == len(p["token_ids"]) == 4096 and p["token_ids"][0] == 2 for p in a)
    assert [p["family"] for p in a] == list(cap.FAMILIES) and all(len(p["files"]) >= 2 for p in a)
    assert all(f["tokens"] <= cap.FILE_TOKENS + 2 and len(f["sha256"]) == 64 for p in a for f in p["files"])
    assert a[0]["token_sha256"] == cap.ids_sha(a[0]["token_ids"])


class Attention(nn.Module):
    """A stand-in with the class name and attributes of the vLLM Attention module (attention.py 333 to 339)."""

    def __init__(self, layer: int, sliding: bool):
        super().__init__()
        self.layer_name = f"language_model.model.layers.{layer}.self_attn.attn"
        self.num_heads, self.num_kv_heads, self.head_size = 32, (16 if sliding else 4), (256 if sliding else 512)
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


def test_hooks_record_the_global_layers(tmp_path: Path) -> None:
    """register() hooks layers 5 and 11 only; a prefill in two chunks gives the last q_rows query rows and every key
    row in position order; the k_norm weights are read; the files have a SHA-256."""
    model, store = fake_model(), cap.Store(q_rows=4)
    reg = cap.register(model, store)
    assert [h["layer"] for h in reg["hooked"]] == [5, 11] and sorted(reg["k_norm"]) == [5, 11]
    assert float(reg["k_norm"][11][0]) == 12.0 and not cap.hook_problems(reg["hooked"], [5, 11])
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
    assert data[5]["src"] == ["positions"]
    files = cap.save_prompt(tmp_path, 0, data)
    assert set(files) == {"5", "11"} and all(len(f["sha256"]) == 64 for f in files.values())
    back = torch.load(tmp_path / "p0" / "L05.pt", weights_only=True)
    assert torch.equal(back["k"], data[5]["k"]) and back["layer"] == 5
    store.begin(10)                                      # without the parent hook: the row order, and a bad shape
    model.language_model.model.layers[5].self_attn.attn(q[:8], k[:8], v[:8])
    bad = cap.check_prompt(store.end(), [5, 11], 10, 4, [])
    assert any("layer 11: no tensors" in b for b in bad) and any("k shape" in b for b in bad)
    assert cap.check_prompt({}, [5, 11], 10, 4, []) == ["no global-layer tensors were recorded"]
    wrong = [dict(h, num_kv_heads=8) for h in reg["hooked"]]
    assert cap.hook_problems(wrong, [5, 11, 17])[0].startswith("hooked layers")


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


def edit_metrics(res: Path, fn) -> dict:
    p = res / "metrics.json"
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d))
    return analyze(res)


def test_pass(tmp_path: Path) -> None:
    s = run_stub(tmp_path)
    assert s["decision"].startswith("PASS (2 global layers"), s["decision"]
    assert all(s["checks"][k] for k in ("V1", "V2", "V3", "V4")) and s["checks"]["_identity_max"] < 3e-3
    for v in s["layers"].values():
        assert 0.8 < v["ratio_med"] < 1.25 and v["w_forms"] == ["weight"] and v["rope_after"] and v["prompts"] == 2
    assert len(s["ledger_metrics"]["ratio"]["values"]) == 4 and s["prompts"][0]["token_sha256"]


def test_module_forms_of_w(tmp_path: Path) -> None:
    """The data decide the form: a module that stores w - 1 gives one_plus_weight; no module weight gives the data."""
    s = run_stub(tmp_path / "one", "--stub-w-form", "one_plus_weight")
    assert s["decision"].startswith("PASS") and all(v["w_forms"] == ["one_plus_weight"] for v in s["layers"].values())
    s = run_stub(tmp_path / "none", "--stub-w-form", "none")
    assert s["decision"].startswith("PASS") and all(v["w_forms"] == ["data"] for v in s["layers"].values())


def test_fail_paths(tmp_path: Path) -> None:
    s = run_stub(tmp_path / "q", "--stub-q-max", "60000")                # |q'| = 60,000 x w, above 32,752
    assert s["decision"].startswith("FAIL (largest |q'|"), s["decision"]
    run_stub(tmp_path / "r")

    def worse(d):                                                        # layer 11: 2.5 x the standard error
        for m in d["prompts"].values():
            m["11"]["err_cmp"] = 2.5 * m["11"]["err_std"]

    s = edit_metrics(tmp_path / "r", worse)
    assert s["decision"].startswith("FAIL (layer 11: median err_compact"), s["decision"]


def test_inconclusive_paths(tmp_path: Path) -> None:
    run_stub(tmp_path)

    def scale(layer: str, f: float, prompts=("0", "1")):
        def fn(d):
            for p in prompts:
                d["prompts"][p][layer]["err_cmp"] = f * d["prompts"][p][layer]["err_std"]
        return fn

    s = edit_metrics(tmp_path, scale("5", 1.5))
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: median ratio 1.500 above 1.25"), s["decision"]
    run_stub(tmp_path)

    def one_bad(d):                                                     # the median holds, the largest does not
        p = d["prompts"]
        top = max(p["0"]["5"]["err_std"], p["1"]["5"]["err_std"])
        p["0"]["5"]["err_cmp"] = 0.1 * p["0"]["5"]["err_std"]
        p["1"]["5"].update(err_std=top, err_cmp=2.1 * top)              # median ratio about 1.1, largest 2.1

    s = edit_metrics(tmp_path, one_bad)
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: largest-error ratio"), s["decision"]
    run_stub(tmp_path)

    def sat(d):                                                         # 0.2% of the compact rows saturated
        for m in d["prompts"].values():
            m["5"]["store_cmp"]["sat"] = int(0.002 * m["5"]["store_cmp"]["n"])

    s = edit_metrics(tmp_path, sat)
    assert s["decision"].startswith("INCONCLUSIVE (layer 5: saturated share"), s["decision"]
    run_stub(tmp_path)

    def big_q(d):
        d["prompts"]["0"]["11"]["qv_max"] = 10000.0

    s = edit_metrics(tmp_path, big_q)
    assert s["decision"].startswith("INCONCLUSIVE (layer 11: largest |q'| 10,000.0 above 6,550.0"), s["decision"]


def test_invalid_paths(tmp_path: Path) -> None:
    s = run_stub(tmp_path / "gpu", "--stub-gpu", "STUB NVIDIA L4")
    assert s["decision"].startswith("INVALID (V1"), s["decision"]
    s = run_stub(tmp_path / "ff", "--stub-no-tensors", check=False)
    assert s["decision"].startswith("INVALID (V2: fail fast after the first prompt: no global-layer tensors")
    s = run_stub(tmp_path / "err", "--stub-fail", check=False)
    assert s["decision"].startswith("INVALID (V2") and "stub failure" in s["decision"], s["decision"]
    s = run_stub(tmp_path / "id", "--stub-break-identity")
    assert s["decision"].startswith("INVALID (V3"), s["decision"]
    s = run_stub(tmp_path / "nan", "--stub-nan")
    assert s["decision"].startswith("INVALID (V4"), s["decision"]
    run_stub(tmp_path / "plan")                                         # a real run must use the frozen plan
    p = tmp_path / "plan" / "capture_run.json"
    p.write_text(json.dumps(json.loads(p.read_text()) | {"stub": False}))
    s = analyze(tmp_path / "plan")
    assert s["decision"].startswith("INVALID (V2") and "frozen plan" in s["decision"], s["decision"]
    run_stub(tmp_path / "miss")
    (tmp_path / "miss" / "cap" / "p1" / "L11.pt").unlink()
    an.compute_metrics(tmp_path / "miss" / "cap", tmp_path / "miss", "cpu")
    s = analyze(tmp_path / "miss")
    assert s["decision"].startswith("INVALID (V2") and "prompt 1 layer 11: no metrics" in s["decision"]
    assert analyze(tmp_path / "empty")["decision"].startswith("INVALID (V1, V2, V3, V4")


def test_timing_at_the_real_size() -> None:
    """S3 (E-38): one prompt and layer at the real size (8,192 tokens, 256 queries) on the Mac CPU, and the hook copy
    of one real-size layer. The H100 runs 40 of them; the README time model uses this bound."""
    w, c = gemma_case(8192, 256, seed=7)
    t = time.time()
    m = an.layer_metrics(c, w.float(), "cpu")
    dt = time.time() - t
    assert dt < 300 and 0.8 < m["err_cmp"] / m["err_std"] < 1.25 and m["identity_err"] < 3e-3, dt
    store = cap.Store(q_rows=256)
    q = torch.zeros(8192, 32 * 512, dtype=torch.bfloat16)
    k = torch.zeros(8192, 4 * 512, dtype=torch.bfloat16)
    store.begin(8192)
    t = time.time()
    store.attn_hook(5, 32, 4, 512)(None, (q, k, k), {})
    data = store.end()
    assert time.time() - t < 30 and list(data[5]["q"].shape) == [256, 32, 512]
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
    session(code, res)
    assert "capture: ok before, skip" in (res / "session.log").read_text()
    session(code, tmp_path / "failed", STUB_ARGS="--stub-fail")
    assert "decision: INVALID (V2" in (tmp_path / "failed" / "session.log").read_text()
    session(code, tmp_path / "ff", STUB_ARGS="--stub-no-tensors")
    log = (tmp_path / "ff" / "session.log").read_text()
    assert "capture: exit 3" in log and "decision: INVALID (V2: fail fast" in log, log


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Tests for E-54 (no GPU, no vLLM): the tied row against the standard storage with the same scales (exact when w is a
power of 2), the kernel form of the tied row against an independent natural-order form, the calibration rule (a
planted phase, the tie-break), the V5 negative controls of R-74 C8, the prompts (no E-53 file, no file twice, chat
markers, the long prompt, special ids, the seed), the hooks over 4 prefill chunks, the kernel arm with the CPU stand-in
(MATCH, and MISMATCH with swapped descales), the frozen-scales path of the analysis, every decision path, the session
caps and the cost, one timing check at the real size, and a stub session from a clean copy of the upload folders.

Run: uv run --no-project --with numpy --with torch --with pytest python -m pytest -q --rootdir <folder>
     <folder>/test_e54.py -p no:cacheprovider
"""

from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_accuracy"))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))

import analyze_e54 as a54  # noqa: E402
import calibrate_e54 as cal  # noqa: E402
import capture_e54 as cap  # noqa: E402
from analyze_e54 import an, ca  # noqa: E402

UPLOAD = ["gemma4_kv_scaletie", "gemma4_kv_accuracy", "gemma4_kv_kernel", "runpod"]   # README Section 11
PY = sys.executable


class Args:
    """The stub flags that capture_e53.stub_layer reads, with their defaults."""
    stub_q_scale, stub_q_max, stub_big_k, stub_big_k_dim = 2.0, 0.0, 0.0, 0
    stub_break_identity = stub_nan = stub_swap_heads = False
    heads, kv_heads = 32, 4

    def __init__(self, **kw):
        self.__dict__.update(kw)


def case(length: int = 512, q_rows: int = 32, w: float = 0.0625, seed: int = 1, **kw):
    """One global layer built as in Gemma 4 (scalar w, BF16 rounding as the engine) as float32 tensors."""
    gen = torch.Generator().manual_seed(seed)
    wt = torch.full((512,), float(torch.tensor(w).to(torch.bfloat16)), dtype=torch.float64)
    c = cap.c53.stub_layer(length, q_rows, wt, gen, Args(**kw))
    return float(wt[0]), c, tuple(c[n].float() for n in ("q", "k", "v")) + (c["qpos"].long(),)


def test_tied_row_equals_the_standard_storage_when_w_is_a_power_of_two() -> None:
    """With w = 2^-4, code(V_unrot / vs) = code(K_unrot / (w vs)) exactly, so TIED equals TIEDSTD up to the sum order;
    with the real w of layer 41 the codes flip in a few percent of the values (R-74 C1) and the errors stay close."""
    w, _, (q, k, v, qpos) = case(w=0.0625)
    vs = 0.0982
    ks = a54.f32(w * vs)
    t, s = a54.tied_arm(q, k, v, qpos, ks, vs), a54.std_arm(q, k, v, qpos, ks, vs)
    assert an.rel(t, s) < 1e-5
    w2, _, (q, k, v, qpos) = case(w=0.06494140625)
    ref = an.attend(q, k, v, qpos)
    ks2 = a54.f32(w2 * vs)
    et, es = an.rel(a54.tied_arm(q, k, v, qpos, ks2, vs), ref), an.rel(a54.std_arm(q, k, v, qpos, ks2, vs), ref)
    assert 0.8 < et / es < 1.25 and an.rel(a54.tied_arm(q, k, v, qpos, ks2, vs), a54.std_arm(q, k, v, qpos, ks2, vs)) > 0


def test_kernel_form_of_the_tied_row_matches_the_natural_order_form() -> None:
    """The views and descales (K view times ks with the plain permuted query) give the attention of the natural-order
    key [ks code(K_rot / ks) | w vs code(V_unrot / vs)] and value vs code(V / vs); also in FP16."""
    w, _, (q, k, v, qpos) = case(w=0.06494140625)
    vs = 0.1801
    ks = a54.f32(w * vs)
    rot, un = torch.tensor(ca.ROT), torch.tensor(ca.UNROT)
    kk = torch.empty_like(k)
    kk[..., rot] = ks * an.fp8(k[..., rot] / ks)
    kk[..., un] = w * vs * an.fp8(v[..., un] / vs)
    vv = vs * an.fp8(v / vs)
    assert an.rel(a54.tied_arm(q, k, v, qpos, ks, vs), an.attend(q, kk, vv, qpos)) < 1e-5
    o16 = an.attend(an.fp16(q), kk, vv, qpos, torch.float16)
    assert an.rel(a54.tied_arm(q, k, v, qpos, ks, vs, fp16=True), o16) < 1e-3


def test_calibration_finds_a_planted_phase_and_breaks_ties_low() -> None:
    """V made exact on the FP8 grid at phase 11 of its base octave: the value-path curve of CALSTD has its minimum
    there; the base is the smallest power of 2 at or above 2 x max / 448; ties go to the lowest phase."""
    assert cal.pick([1.0, 0.5, 0.5, 2.0]) == 1 and cal.pick([0.3, 0.9, 0.3]) == 0
    assert cal.base_scale(22.0) == 0.125 and cal.base_scale(1.421875) == 2.0 ** -7
    gen = torch.Generator().manual_seed(3)
    grid = torch.tensor(sorted({float(x) for x in torch.arange(-1000, 1000).float().div(5).clamp(-160, 160)
                                .to(torch.float8_e4m3fn).float()}))
    b, j_star = 0.125, 11
    s_star = cal.phase_scale(b, j_star)
    caps = []
    for _ in range(2):
        w, _, (q, k, _, qpos) = case(length=256, q_rows=16, w=0.0625, seed=int(torch.randint(100, (1,), generator=gen)))
        codes = grid[torch.randint(len(grid), (256, 4, 512), generator=gen)]
        codes[0, 0, 0] = 160.0                                           # the largest value maps to 160 at s_star
        caps.append((q, k, codes * s_star, qpos))
    assert cal.base_scale(160.0 * s_star) == b
    sc = cal.layer_calibration(caps, 0.0625, phases=32)
    assert sc["calstd"]["phase_v"] == j_star and min(sc["curves"]["calstd_value"]) < 1e-6
    assert sc["tied"]["ks"] == a54.f32(0.0625 * sc["tied"]["vs"])


def test_v5_negative_controls_of_r74() -> None:
    """(R-74 C8) The reference against a BF16 engine output passes 0.005; a wrong head mapping, no causal mask, the
    scale 1/sqrt(512) or query positions that hide the current key fail it (R-74: 0.22 to 1.17 on real data)."""
    _, c, (q, k, v, qpos) = case(length=256, q_rows=32, w=0.0625, seed=4, stub_q_scale=8.0)
    q[..., :] = q + 4.0 * k[qpos].repeat_interleave(8, dim=1)          # a strong self-attention, as real data
    ref = an.attend(q, k, v, qpos)
    eng = ref.to(torch.bfloat16).float()
    assert an.rel(ref, eng) < a54.ENGINE_TOL
    wrong = {"heads": an.attend(q, k.roll(1, dims=1), v.roll(1, dims=1), qpos),
             "no mask": an.attend(q, k, v, torch.full_like(qpos, k.shape[0] - 1)),
             "scale": an.attend(q / math.sqrt(512), k, v, qpos), "positions": an.attend(q, k, v, qpos - 1)}
    for name, o in wrong.items():
        assert an.rel(o, eng) > a54.ENGINE_TOL, name


def text_tree(root: Path, n: int = 40) -> list[Path]:
    """n deterministic text files of 3 KB (half .py, half .md)."""
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        f = root / (f"m{i:02d}.py" if i % 2 else f"d{i:02d}.md")
        f.write_text("".join(f"line {i} {j} value = {i * j}\n" for j in range(150)))
        out.append(f)
    return out


def small_plan() -> dict:
    return dict(a54.PLAN, q_rows=8, prompts=[{"id": 0, "kind": "plain", "family": "vllm_py", "tokens": 300},
                                            {"id": 1, "kind": "plain", "family": "docs", "tokens": 300},
                                            {"id": 2, "kind": "chat", "family": "mixed", "tokens": 300},
                                            {"id": 3, "kind": "plain", "family": "mixed", "tokens": 600}])


def test_prompts_exclude_e53_files_and_hold_chat_markers(tmp_path: Path) -> None:
    files = text_tree(tmp_path / "t")
    (tmp_path / "t" / "z_special.py").write_text("x = '<start_of_turn>'\n" * 200)
    shutil.copy(files[3], tmp_path / "t" / "z_copy.py")                 # an E-53 file under another name
    src = cap.c53.sources({"vllm_py": [tmp_path / "t"], "torch_py": [tmp_path / "t"], "docs": [tmp_path / "t"]})
    ex = {"paths": {"m01.py", "d00.md"}, "sha256": {cap.c53.sha(files[3].read_bytes())}}
    special = frozenset(cap.TOY_SPECIAL.values())
    build = lambda: cap.build_prompts(src, cap.toy_encode, 2, tmp_path / "t", special, cap.toy_chat, ex,  # noqa: E731
                                      small_plan(), chat_chars=120)
    ps = build()
    names = [f["path"] for p in ps for f in p["files"]]
    assert len(names) == len(set(names)) and not {"m01.py", "d00.md", "m03.py", "z_copy.py"} & set(names)
    assert [p["n_tokens"] for p in ps] == [300, 300, 300, 600] and all(p["bos_at_0"] for p in ps)
    assert [p["special_after_0"] for p in ps if p["kind"] == "plain"] == [0, 0, 0]
    chat = ps[2]
    assert chat["special_after_0"] >= 2 and chat["turns"] >= 2 and chat["token_ids"][1] == 1
    assert [p["token_sha256"] for p in build()] == [p["token_sha256"] for p in ps]
    one = dict(small_plan(), prompts=[{"id": 0, "kind": "plain", "family": "vllm_py", "tokens": 70000}])
    p = cap.build_prompts(src, cap.toy_encode, 2, tmp_path / "t", special, cap.toy_chat, ex, one)[0]
    assert p["excluded"] >= 3 and p["skipped_special"] == 1 and p["n_tokens"] == 70000    # every .py file visited
    assert not {"m01.py", "m03.py", "z_copy.py", "z_special.py"} & {f["path"] for f in p["files"]}
    assert cap.toy_encode("<bos>ab<end_of_turn>") == [2, ord("a") + 3, ord("b") + 3, 0]


def test_hooks_record_a_prefill_in_four_chunks() -> None:
    """The E-53 Store over 4 chunks (the 32K prompt): the last q_rows query rows, the output rows of the same
    positions and every key row in position order."""
    store = cap.c53.Store(q_rows=4)
    gen = torch.Generator().manual_seed(6)
    q = torch.randn(16, 2 * 512, generator=gen).to(torch.bfloat16)
    k, v = (torch.randn(16, 512, generator=gen).to(torch.bfloat16) for _ in range(2))
    pre, attn, post = store.pos_hook(5), store.attn_hook(5, 2, 1, 512), store.out_hook(5, 2, 512)
    store.begin(16)
    for lo in range(0, 16, 4):
        sl = slice(lo, lo + 4)
        pre(None, (), {"positions": torch.arange(lo, lo + 4)})
        attn(None, (q[sl], k[sl], v[sl]), {})
        post(None, (q[sl], k[sl], v[sl]), q[sl])
    assert len(store.parts[5]["k"]) == 4
    data = store.end()
    assert cap.c53.check_prompt(data, [5], 16, 4, [], (2, 1, 512)) == []
    assert torch.equal(data[5]["q"], q[12:].reshape(4, 2, 512)) and torch.equal(data[5]["k"], k.reshape(16, 1, 512))


def run_stub(res: Path, *extra: str) -> dict:
    r = subprocess.run([PY, str(HERE / "capture_e54.py"), "--stub", "--out", str(res), "--capture", str(res / "cap"),
                        *extra], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    a54.compute_metrics(res / "cap", res, "cpu", None)
    return analyze(res)


def analyze(res: Path) -> dict:
    s = a54.summarize(res)
    s["decision"] = a54.decide(s)
    return s


def edit_json(path: Path, fn) -> None:
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


@pytest.fixture(scope="module")
def stub_res(tmp_path_factory) -> Path:
    res = tmp_path_factory.mktemp("stub") / "e54"
    s = run_stub(res)
    assert s["decision"].startswith("STUB PASS ("), s["decision"]
    return res


def variant(stub_res: Path, tmp_path: Path, fn_met=None, fn_prompts=None, fn_run=None) -> dict:
    res = tmp_path / "v"
    shutil.copytree(stub_res, res, ignore=shutil.ignore_patterns("cap"))
    for name, fn in (("metrics.json", fn_met), ("prompts.json", fn_prompts), ("capture_run.json", fn_run)):
        if fn:
            edit_json(res / name, fn)
    return analyze(res)


def scale_arms(m: dict, f: float, arms=("TIED", "TIEDSTD", "TIED_E5M2"), key: str = "err", pid=None, layer=None):
    for p, layers in m["prompts"].items():
        for lyr, cell in layers.items():
            if (pid is None or p == pid) and (layer is None or lyr == layer):
                for a in arms:
                    cell[key][a] *= f


def test_decision_paths(stub_res: Path, tmp_path: Path) -> None:
    base = analyze(stub_res)
    st = base["stats"]
    r = max(st["R1"], st["R2"])
    assert st["tie"] == pytest.approx(1.0, abs=1e-4) and st["e5m2"] > a54.E5M2_MIN
    s = variant(stub_res, tmp_path, lambda m: scale_arms(m, 1.6 / r))
    assert s["decision"].startswith("STUB FAIL (R") and "above 1.5" in s["decision"], s["decision"]
    shutil.rmtree(tmp_path / "v")
    s = variant(stub_res, tmp_path, lambda m: scale_arms(m, 1.4 / r))
    assert s["decision"].startswith("STUB INCONCLUSIVE (") and "above 1.25" in s["decision"], s["decision"]
    shutil.rmtree(tmp_path / "v")
    s = variant(stub_res, tmp_path, lambda m: scale_arms(m, 1.8, pid="6"))           # one prompt only
    assert s["decision"].startswith("STUB INCONCLUSIVE (prompt 6: R1"), s["decision"]
    shutil.rmtree(tmp_path / "v")

    def one_layer(m):                                     # layer 5 small and 2.5 times worse; layer 11 decides P
        for layers in m["prompts"].values():
            c = layers["5"]["err"]
            c.update(STOCK=1e-3, CALSTD=1e-3, TIED=2.5e-3, TIEDSTD=2.5e-3, TIED_E5M2=6e-3)
    s = variant(stub_res, tmp_path, one_layer)
    assert "layer 5: median TIED/STOCK 2.500 above 2.0" in s["decision"] and "INCONCLUSIVE" in s["decision"]
    shutil.rmtree(tmp_path / "v")
    s = variant(stub_res, tmp_path, lambda m: scale_arms(m, 1.4 / st["R1_fp16"], arms=("TIED",), key="err16"))
    assert "R1_fp16" in s["decision"] and "INCONCLUSIVE" in s["decision"], s["decision"]
    shutil.rmtree(tmp_path / "v")
    s = variant(stub_res, tmp_path, lambda m: m["prompts"]["0"]["5"].update(nonfinite_fp16=3))
    assert "3 FP16 values not finite" in s["decision"]
    shutil.rmtree(tmp_path / "v")


def test_invalid_paths(stub_res: Path, tmp_path: Path) -> None:
    cases = {
        "V5": (lambda m: m["prompts"]["2"]["11"].update(engine_gap=0.006), None, None),
        "V6": (lambda m: scale_arms(m, 0.6, arms=("TIED_E5M2",)), None, None),
        "V7": (lambda m: scale_arms(m, 1.2, arms=("TIED", "TIED_E5M2")), None, None),
        "V3": (lambda m: m["prompts"]["1"]["5"].update(w_value=0.0626), None, None),
        "V1": (None, None, lambda r: r["gpu"].update(name="NVIDIA GeForce RTX 4090")),
        "V2": (None, lambda p: p["prompts"][0]["files"].append(
            {"path": "vllm/models/deepseek_v32/nvidia/ops/fused_q_cutedsl.py", "sha256": "x"}), None),
    }
    for want, (fm, fp, fr) in cases.items():
        s = variant(stub_res, tmp_path, fm, fp, fr)
        assert s["decision"].startswith(f"STUB INVALID ({want}"), (want, s["decision"])
        if want == "V5":
            assert "no new Pod" in s["decision"]
        shutil.rmtree(tmp_path / "v")
    s = variant(stub_res, tmp_path, fn_prompts=lambda p: p["prompts"][4].update(special_after_0=1))
    assert "chat prompt 4" in s["decision"] and "INVALID (V2" in s["decision"]
    shutil.rmtree(tmp_path / "v")
    s = variant(stub_res, tmp_path, fn_prompts=lambda p: p["prompts"][1].update(special_after_0=2))
    assert "prompt 1 has 2 special ids" in s["decision"]


def test_kernel_arm_with_the_cpu_standin(stub_res: Path, tmp_path: Path) -> None:
    """The caches, views, block table and descales of the kernel arm give the emulation (MATCH); swapped descales give a
    gap near 1 (MISMATCH); the analysis prints the kernel line."""
    res = tmp_path / "k"
    shutil.copytree(stub_res, res)
    run = lambda *x: subprocess.run([PY, str(HERE / "kernel_e54.py"), "--res", str(res), "--capture",  # noqa: E731
                                     str(res / "cap"), "--standin", *x], capture_output=True, text=True, timeout=600)
    r = run()
    assert r.returncode == 0, r.stderr[-2000:]
    k = json.loads((res / "kernel.json").read_text())
    assert k["outcome"] == "MATCH" and max(k["max_gap"].values()) < a54.ENGINE_TOL and len(k["cells"]) == 4, k
    assert "kernel arm (secondary): MATCH" in analyze(res)["decision"]
    run("--swap-descales")
    k = json.loads((res / "kernel.json").read_text())
    assert k["outcome"] == "MISMATCH" and k["max_gap"]["STOCK"] < a54.ENGINE_TOL < k["max_gap"]["TIED"], k
    r = subprocess.run([PY, str(HERE / "kernel_e54.py"), "--res", str(res), "--capture", str(res / "cap")],
                       capture_output=True, text=True, timeout=300)
    if not torch.cuda.is_available():
        assert json.loads((res / "kernel.json").read_text())["outcome"] == "NOT RUN"


def test_frozen_scales_path(stub_res: Path, tmp_path: Path) -> None:
    """calibrate_e54 on the stub captures writes a scales file with the SHA-256 of its inputs; the analysis of a run
    that is not a stub reads that file, records its SHA-256 and uses its scales."""
    out = tmp_path / "scales.json"
    r = subprocess.run([PY, str(HERE / "calibrate_e54.py"), "--capture", str(stub_res / "cap"), "--run",
                        str(stub_res / "capture_run.json"), "--out", str(out), "--phases", "8"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    sc = json.loads(out.read_text())
    assert sorted(sc["layers"]) == ["11", "5"] and len(sc["inputs"]["files"]) == 14 and sc["in_sample"]["tie"] > 0.9
    res = tmp_path / "f"
    shutil.copytree(stub_res, res)
    edit_json(res / "capture_run.json", lambda d: d.update(stub=False))
    met = a54.compute_metrics(res / "cap", res, "cpu", out)
    assert met["scales_source"] == "scales.json" and met["scales_sha256"] == a54.sha_file(out)
    assert met["scales"]["5"]["tied"] == sc["layers"]["5"]["tied"]
    assert "INVALID (V2" in analyze(res)["decision"]                    # a stub capture is not the frozen plan


def session_defaults() -> dict:
    text = (HERE / "session_e54.sh").read_text()
    return {k: int(v) for k, v in re.findall(r"^(\w+)=\$\{\w+:-(\d+)\}", text, re.M)}


def test_session_caps_keep_the_watchdog_order_and_the_cost_cap() -> None:
    """The step caps fit inside the session watchdog; the session watchdog fires before the boot watchdog (the session
    starts about 140 s after boot); the worst case (a 15 min pull, then the boot watchdog) stays inside the $4 cap at
    $3.99/h, and the S5 rehearsal inside its $0.60 cap (RTX 4090 at $0.89/h; POD_LIMIT 1,200 s, boot cap 1,500 s)."""
    d = session_defaults()
    assert (d["DL_TIMEOUT"], d["CAP_TIMEOUT"], d["AN_TIMEOUT"], d["KER_TIMEOUT"], d["POD_LIMIT"],
            d["AUTO_STOP_DELAY"]) == (420, 720, 360, 300, 1920, 450)
    boot, start, setup, gpu_wait = 2400, 140, 60, 60
    assert d["DL_TIMEOUT"] + d["CAP_TIMEOUT"] + d["AN_TIMEOUT"] + d["KER_TIMEOUT"] + setup + gpu_wait <= d["POD_LIMIT"]
    assert start + d["POD_LIMIT"] < boot
    assert (15 * 60 + boot) / 3600 * 3.99 <= 4.00                        # 55 min, $3.66
    assert (15 * 60 + 1500) / 3600 * 0.89 <= 0.60 and start + 1200 < 1500


def test_timing_at_the_real_size() -> None:
    """S3 (E-38): one prompt and layer at 8,192 tokens and 256 queries with every arm of README Section 5 (16 attention
    passes) on the Mac CPU. The H100 runs 60 such cells and one 32K prompt (about 4 times the work per layer)."""
    w, c, _ = case(8192, 256, w=0.06494140625, seed=7)
    sc = {"w": w, "calstd": {"ks": 2.0 ** -6, "vs": 0.125}, "tied": {"ks": a54.f32(w * 0.125), "vs": 0.125}}
    t = time.time()
    m = a54.layer_metrics(c, torch.full((512,), w), sc, "cpu")
    dt = time.time() - t
    assert dt < 600 and m["engine_gap"] < a54.ENGINE_TOL and m["w_scalar"] and m["identity_err"] < 3e-3, dt
    assert 0.8 < m["err"]["TIED"] / m["err"]["TIEDSTD"] < 1.25 and m["err"]["TIED_E5M2"] > 1.5 * m["err"]["TIED"]
    print(f"layer_metrics at the real size: {dt:.1f} s on the CPU")


def clean_copy(tmp_path: Path) -> Path:
    code = tmp_path / "code"
    for sub in UPLOAD:
        shutil.copytree(HERE.parent / sub, code / sub,
                        ignore=shutil.ignore_patterns("results", "exploratory", "dryrun", "__pycache__", "*.gz",
                                                      ".pytest_cache"))
    return code


def session(code: Path, res: Path, **extra: str) -> subprocess.CompletedProcess:
    e = dict(os.environ, STUB="1", RES=str(res), CAPDIR=str(res.parent / (res.name + "_cap")), CODE=str(code),
             AUTO_STOP="0", PATH=f"{Path(PY).parent}:{os.environ.get('PATH', '')}", **extra)
    return subprocess.run([shutil.which("sh"), str(code / "gemma4_kv_scaletie" / "session_e54.sh")], env=e,
                          capture_output=True, text=True, timeout=900)


def test_stub_session(tmp_path: Path) -> None:
    code, res = clean_copy(tmp_path), tmp_path / "e54"
    r = session(code, res)
    assert r.returncode == 0, r.stderr[-2000:]
    log = (res / "session.log").read_text()
    assert "capture: exit 0" in log and "analysis: exit 0" in log and "kernel arm: exit 0" in log, log
    assert "decision: STUB PASS (" in log and "kernel arm (secondary): MATCH" in log, log
    assert (res / "summary.json").exists() and (res / "kernel.json").exists() and not list(res.glob("**/*.pt"))
    shas = [line.split()[1] for line in (res / "capture_sha.txt").read_text().split("\n") if line]
    assert len(shas) == 15 and "k_norm.pt" in shas and "p6/L11.pt" in shas and "captures: 15 files" in log
    session(code, res)
    assert "capture: ok before, skip" in (res / "session.log").read_text()
    session(code, tmp_path / "failed", STUB_ARGS="--stub-fail")
    assert "decision: STUB INVALID (V2" in (tmp_path / "failed" / "session.log").read_text()
    session(code, tmp_path / "ff", STUB_ARGS="--stub-no-tensors")
    log = (tmp_path / "ff" / "session.log").read_text()
    assert "capture: exit 3" in log and "decision: STUB INVALID (V2: fail fast" in log, log


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

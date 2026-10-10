"""E-27 pre-run checks (unit tests). Run: python test_vcache.py [--model]
--model also runs the checks that load Qwen3-0.6B (T4 and T7)."""

import math
import sys

import torch

import vcache as vc

TEXTS = __import__("os").path.expanduser("~/Downloads/e27_data/e27_texts.json")

torch.manual_seed(0)


class FakeModule:
    layer_idx = 0


def t1_quantizer_error():
    u = torch.randn(50, 4, 128)
    q, step = vc.quantize_groups(u, 2)
    assert ((q - u).abs() <= step / 2 + 1e-6).all()
    q3, step3 = vc.quantize_groups(u, 3)
    assert ((q3 - u).abs() <= step3 / 2 + 1e-6).all()
    print("T1 quantizer error <= step/2: PASS")


def t2_carry_bounded_and_telescoping():
    v = torch.randn(400, 3, 64) * torch.linspace(0.2, 3, 400).view(-1, 1, 1)
    vh = vc.quantize_sequence(v, 2, rotated=False, shaping=True)
    err = (vh - v)[vc.SINK:]
    cum = err.cumsum(0)                                   # = -carry after each token
    _, step = vc.quantize_groups(v[vc.SINK:], 2)
    # carry is bounded by half of the step of the token that produced it; the step is computed on
    # u = v + carry, so allow the largest step seen
    bound = step.amax() / 2 * 1.5
    assert cum.abs().max() <= bound, (float(cum.abs().max()), float(bound))
    plain = vc.quantize_sequence(v, 2, rotated=False, shaping=False)
    assert (plain - v)[vc.SINK:].cumsum(0).abs().max() > 10 * cum.abs().max()
    print(f"T2 carry bounded (max |cum err| {float(cum.abs().max()):.3f} <= {float(bound):.3f}), "
          f"plain cum err much larger: PASS")


def t3_hadamard():
    h = vc.hadamard(64)
    assert torch.allclose(h @ h.T, torch.eye(64), atol=1e-6)
    v = torch.randn(10, 2, 128)
    assert torch.allclose(vc.rotate(vc.rotate(v), inverse=True), v, atol=1e-5)
    print("T3 Hadamard orthonormal, rotate/inverse identity: PASS")


def explicit_attention(q, k, v, vhat, scaling, sink=vc.SINK, recent=vc.RECENT, self_q=0):
    hq, T, D = q.shape
    rep = hq // k.shape[0]
    kk = k.repeat_interleave(rep, 0)
    vv = v.repeat_interleave(rep, 0)
    vq = vhat.repeat_interleave(rep, 0)
    out = torch.empty(hq, T, D)
    for t in range(T):
        scores = torch.einsum("hd,hjd->hj", q[:, t], kk[:, :t + 1]) * scaling
        p = torch.softmax(scores, -1)
        j = torch.arange(t + 1)
        m = ((j >= sink) & (j < t - recent + self_q)).view(1, -1, 1)
        vt = torch.where(m, vq[:, :t + 1], vv[:, :t + 1])
        out[:, t] = torch.einsum("hj,hjd->hd", p, vt)
    return out


def t5_masked_correction():
    hq, hkv, T, D = 4, 2, 520, 64
    q = torch.randn(1, hq, T, D)
    k = torch.randn(1, hkv, T, D)
    v = torch.randn(1, hkv, T, D)
    vc.STATE.mode, vc.STATE.cond, vc.STATE.chunk = "quant", dict(bits=2, rotated=True, shaping=True), 100
    out, _ = vc.attention(FakeModule(), q, k, v, None, 1 / math.sqrt(D))
    vhat = vc.quantize_sequence(v[0].transpose(0, 1), 2, rotated=True, shaping=True).transpose(0, 1)
    ref = explicit_attention(q[0], k[0], v[0], vhat, 1 / math.sqrt(D))
    diff = (out[0].transpose(0, 1) - ref).abs().max()
    assert diff < 1e-4, float(diff)
    vc.STATE.mode, vc.STATE.chunk = "fp", 256
    print(f"T5 masked correction = explicit per-query construction (max diff {float(diff):.2e}): PASS")


def t6_g_formula():
    T, H, D, trials = 600, 1, 64, 300
    b = T
    cases = {
        "smooth": torch.exp(-torch.arange(T).float().flip(0) / 120.0),
        "uniform": torch.ones(T),
        "spiky": torch.zeros(T).index_fill_(0, torch.tensor([150, 300, 450]), 1.0) + 1e-4,
    }
    for name, w in cases.items():
        w = w.clone()
        w[:vc.SINK] = 0
        w = w / w.sum()
        num, den = vc.smoothness_terms(w, b)
        g2 = float(num / den)
        es = ep = 0.0
        for _ in range(trials):
            v = torch.randn(T, H, D)
            ds = (vc.quantize_sequence(v, 2, rotated=False, shaping=True) - v)[:, 0]
            dp = (vc.quantize_sequence(v, 2, rotated=False, shaping=False) - v)[:, 0]
            es += float(((w.view(-1, 1) * ds).sum(0) ** 2).sum())
            ep += float(((w.view(-1, 1) * dp).sum(0) ** 2).sum())
        mc = es / ep
        ok = 0.7 < mc / g2 < 1.3 if g2 > 0.05 else mc < 0.15
        print(f"T6 {name}: G^2 formula {g2:.4f}, Monte Carlo shaped/plain {mc:.4f} -> {'PASS' if ok else 'FAIL'}")
        assert ok


def t8_measure_consistency():
    hq, hkv, T, D = 2, 1, 700, 64
    q = torch.randn(1, hq, T, D)
    k = torch.randn(1, hkv, T, D)
    v = torch.randn(1, hkv, T, D)
    cond = dict(bits=2, rotated=True, shaping=True)
    rec = vc.new_record(1, hq, ["X"])
    vc.STATE.mode, vc.STATE.measure_conds, vc.STATE.record = "measure", {"X": cond}, rec
    vc.attention(FakeModule(), q, k, v, None, 1 / math.sqrt(D))
    vc.STATE.mode = "fp"
    vhat = vc.quantize_sequence(v[0].transpose(0, 1), 2, rotated=True, shaping=True).transpose(0, 1)
    fp = explicit_attention(q[0], k[0], v[0], v[0], 1 / math.sqrt(D))
    qu = explicit_attention(q[0], k[0], v[0], vhat, 1 / math.sqrt(D))
    ee = 0.0
    n = 0
    for t in range(T):
        if t - vc.RECENT - vc.SINK >= 256 and t % 16 == 0:
            ee += float(((qu[:, t] - fp[:, t]) ** 2).sum())
            n += 1
    got = float(rec["err_energy"]["X"].sum())
    assert abs(got - ee) / ee < 1e-3 and int(rec["n_pos"][0]) == n, (got, ee)
    print(f"T8 measure-mode error energy = explicit ({got:.4f} vs {ee:.4f}, {n} positions): PASS")


def t9_dither_and_orders():
    v = torch.randn(300, 2, 128)
    k = torch.randn(300, 2, 128)
    for kw in (dict(dither=True), dict(shaping=True, order="random"), dict(shaping=True, order="keys", keys=k),
               dict(shaping=True, dither=True)):
        vh = vc.quantize_sequence(v, 16, rotated=True, **kw)
        assert (vh - v).abs().max() < 1e-3, kw
    # 2-bit dither error is bounded in the rotated, sign-flipped domain
    vh = vc.quantize_sequence(v, 2, rotated=True, dither=True)
    s = vc._signs(300, 2, 128, 0)
    x, xh = vc.rotate(v * s), vc.rotate(vh * s)
    _, step = vc.quantize_groups(x[vc.SINK:], 2)
    assert ((xh - x)[vc.SINK:].abs() <= step / 2 + 1e-5).all()
    print("T9 dither, random-order and key-order shaping quantize every token once and invert exactly: PASS")


def t10_all_quantized_variant():
    hq, hkv, T, D = 2, 1, 300, 64
    q, k, v = torch.randn(1, hq, T, D), torch.randn(1, hkv, T, D), torch.randn(1, hkv, T, D)
    cond = dict(bits=2, rotated=True, shaping=True, sink=0, recent=0, self_quantized=True)
    vc.STATE.mode, vc.STATE.cond, vc.STATE.chunk = "quant", cond, 64
    out, _ = vc.attention(FakeModule(), q, k, v, None, 1 / math.sqrt(D))
    vc.STATE.mode, vc.STATE.chunk = "fp", 256
    vhat = vc.quantize_sequence(v[0].transpose(0, 1), 2, rotated=True, shaping=True, sink=0).transpose(0, 1)
    ref = explicit_attention(q[0], k[0], v[0], vhat, 1 / math.sqrt(D), sink=0, recent=0, self_q=1)
    diff = (out[0].transpose(0, 1) - ref).abs().max()
    assert diff < 1e-4, float(diff)
    print(f"T10 all-quantized variant (sink 0, recent 0, own token quantized) = explicit construction (max diff {float(diff):.2e}): PASS")


def t11_causality():
    hq, hkv, T, D = 4, 2, 520, 64
    q, k, v = torch.randn(1, hq, T, D), torch.randn(1, hkv, T, D), torch.randn(1, hkv, T, D)
    vc.STATE.gates = {"G": torch.tensor([[True, False]])}
    for cond in (dict(bits=2, rotated=True, shaping=True), dict(bits=2, rotated=True, shaping=True, order="keys"),
                 dict(bits=2, rotated=True, shaping=True, order="random", gate="G"),
                 dict(bits=2, rotated=True, dither=True)):
        vc.STATE.mode, vc.STATE.cond = "quant", cond
        full, _ = vc.attention(FakeModule(), q, k, v, None, 0.125)
        for cut in (250, 300, 400, 517):
            part, _ = vc.attention(FakeModule(), q[:, :, :cut], k[:, :, :cut], v[:, :, :cut], None, 0.125)
            assert (full[0, :cut] - part[0]).abs().max() == 0, (cond, cut)
    vc.STATE.mode = "fp"
    print("T11 causality: truncating the sequence changes no earlier output in any mode: PASS")


def t12_real_v_assumption(model, tok):
    """Report only: plain rounding errors of real V are not independent across tokens."""
    cap = {}
    orig = vc.attention
    def grab(module, query, key, value, *a, **k):
        if module.layer_idx in (0, 14):
            cap[module.layer_idx] = value[0].transpose(0, 1).clone()
        return orig(module, query, key, value, *a, **k)
    from transformers import AttentionInterface
    AttentionInterface.register("vcache", grab)
    text = open(__file__).read() * 4
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"][:720]])
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    AttentionInterface.register("vcache", orig)
    for layer, v in cap.items():
        T = v.shape[0]
        w = torch.zeros(T)
        w[vc.SINK:T] = 1.0 / (T - vc.SINK)
        for name, kw in (("plain", dict()), ("shaped", dict(shaping=True))):
            dv = vc.quantize_sequence(v, 2, rotated=True, **kw) - v
            actual = float(((w.view(-1, 1, 1) * dv).sum(0) ** 2).sum())
            indep = float(((w.view(-1, 1, 1) ** 2) * dv ** 2).sum())
            print(f"T12 layer {layer} {name}: actual / independent-error energy = {actual / indep:.2f} (report only)")


def t4_t7_model():
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
    import e27_run
    AttentionInterface.register("vcache", vc.attention)
    tok = AutoTokenizer.from_pretrained(e27_run.MODEL)
    ref = AutoModelForCausalLM.from_pretrained(e27_run.MODEL, dtype=torch.float32, attn_implementation="eager").eval()
    mod = AutoModelForCausalLM.from_pretrained(e27_run.MODEL, dtype=torch.float32, attn_implementation="vcache").eval()
    text = open(__file__).read() * 3
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"][:700]])
    vc.STATE.mode = "fp"
    with torch.no_grad():
        a = ref(input_ids=ids, use_cache=False).logits
        b = mod(input_ids=ids, use_cache=False).logits
    diff = float((a - b).abs().max())
    assert diff < 2e-3, diff
    print(f"T4 custom attention (fp) reproduces eager logits (max diff {diff:.2e}): PASS")
    wins = e27_run.windows(tok, TEXTS)
    prompts = e27_run.passkey_prompts(tok, wins["haystack"], n=6)
    ok = 0
    for pr in prompts:
        with torch.no_grad():
            lg = mod(input_ids=torch.tensor([pr["prompt"] + pr["answer"]]), use_cache=False).logits[0]
        L = len(pr["prompt"])
        ok += lg[L - 1:L - 1 + len(pr["answer"])].argmax(-1).tolist() == pr["answer"]
    print(f"T7 passkey sanity, full precision: {ok}/6 correct (prompt length {len(prompts[0]['prompt'])} tokens)")
    t12_real_v_assumption(mod, tok)


if __name__ == "__main__":
    t1_quantizer_error()
    t2_carry_bounded_and_telescoping()
    t3_hadamard()
    t5_masked_correction()
    t6_g_formula()
    t8_measure_consistency()
    t9_dither_and_orders()
    t10_all_quantized_variant()
    t11_causality()
    if "--model" in sys.argv:
        t4_t7_model()

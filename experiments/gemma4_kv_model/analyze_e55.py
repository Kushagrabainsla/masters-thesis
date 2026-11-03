"""E-55 analysis (README Sections 5 to 7, amendment A1): the statistics, the validity checks and the frozen decision
rule, from the files of one run (`run_main.json`, `arrays/*.npz`, `gen_outputs.json`, `code_sha.txt`,
`code_manifest.json`). `preflight_problems` is also the pre-flight check of `run_e55.py`.

Statistic (README Section 5): for an arm X and a document d, S_X(d) is the sum over the scored tokens (index t >= 1025:
the context holds more than the 1,024-token sliding window) of the BF16 log-probability minus the arm's log-probability
(the NLL change), and n(d) the number of scored tokens. Delta_X = sum_d S_X(d) / sum_d n(d), in nats per token. The
decision interval (A1, R-77 F5) is a stratified t interval over documents: the variance of the linearized ratio comes
from the spread of the residuals S_X(d) - Delta_X n(d) inside each domain (4 books, 4 prose, 4 code), with Satterthwaite
degrees of freedom. D = Delta_TIED8 - Delta_STOCK8 and E = Delta_TIED8 - 1.5 Delta_STOCK8 use the same method on the
paired document sums. Reported: the percentile bootstrap over documents (10,000 draws, seed 55; R-77 measured 86%
coverage with 4 documents per domain) and a bootstrap over 2,048-token blocks.

Usage: analyze_e55.py --res <dir> [--data data_e55.json] [--manifest-out <file> --design-commit <sha>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_SHA256 = "8f2e2ecd96402b51c13d2f73f25cfe6e0a10c7b6c07a97d0925a334441df36e4"   # data_e55.json (V2; frozen)
SCALES_SHA256 = "8efe6e9076e28e011ff560106665998434b9403a1ddd9f653bdd98e5c718bd43"   # scales_e54.json (E-54 A1)
REVISION = "d4ab4f579dd3516f97d8a6a4c98d0653480bad15"
GLOBAL_LAYERS = (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
HEADS = (32, 4, 512)
DOMAINS = ("books", "prose", "code")
HOOKED = ("BF16", "BF16C", "E5M2", "STOCK8", "TIED8", "BF16R", "JOINT8", "TIED8_UP")   # = splice_e55.ARMS
SCORED = HOOKED[1:]
DECISION = ("BF16", "BF16C", "E5M2", "STOCK8", "TIED8")
SCORE_FROM = 1_025                      # the first scored token index (context of 1,025 tokens or more)
BINS = ((1_025, 8_192), (8_192, 32_768), (32_768, 65_536))
BOOT, SEED = 10_000, 55
M16, M16_REL, KILL16 = 0.001, 0.1, 0.01              # H16: the fixed margin (A1, R-77 F4); 0.1 Delta_E5M2 reported
TOL8_ABS, TOL8_REL, KILL8_RATIO = 0.0002, 0.25, 1.5  # H8: tolerance max(0.0002, 0.25 Delta_STOCK8); KILL8 on E
ENGINE_TOKENS = 4_096
GAP_TOL = 0.01                          # V5: the engine output against the float32 reference of the arm (E-54 0.0017)
SEP_MIN, SEP_LAYERS_BF16C = 0.001, 5    # V7: a layer decides if its two references differ by more than SEP_MIN
DET_TOL = 1e-4                          # V3: a rerun that is not bit-identical may differ by this much (M16 / 10)
FAR_N, FAR_GAP = 12, 1_025              # the far-repeat subset (reported, A1, R-77 O2)
V_KEYS = ("V1", "V2", "V3", "V4", "V5", "V7")
CODE_FILES = ("splice_e55.py", "run_e55.py", "analyze_e55.py", "session_e55.sh", "data_e55.json", "scales_e54.json")
ACTION = {
    "GO16": "the thesis and the paper claim that the compact BF16 cache keeps the model's next-token predictions "
            "within 0.001 nats per token (model level, 3 domains, 32K to 64K tokens); the BF16 compact build (D-88) "
            "can start, then the engine capacity test",
    "PIVOT16": "the compact BF16 cache has a small measured cost: report it; the user decides on the build",
    "KILL16": "stop the 16-bit claim and find the cause (a fault is likely, because the layout is exact in real "
              "arithmetic) before any new run",
    "CHECK16": "a significant improvement cannot come from rounding: treat it as an artifact, find the cause, no claim",
    "INCONCLUSIVE16": "report; the user decides on a second run with more documents (README Section 9: about four "
                      "times the tokens halves the interval) or another metric (a new S4)",
    "GO8": "the compact FP8 row stays a build option after the BF16 build (not worse than the stock FP8 cache by more "
           "than the tolerance)",
    "PIVOT8": "the compact FP8 row is measurably worse than the stock FP8 cache: it needs new scales (R-76 offline "
              "work) and a new test before a build",
    "KILL8": "no compact FP8 claim above the attention level; the build uses the BF16 compact cache only",
    "INCONCLUSIVE8": "report; no compact FP8 claim at the model level",
    "INVALID": "fix the cause; a rerun needs a new S4",
}


def load(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def doc_sums(base: dict, arm: dict, doc_ids: list[str], lo: int = SCORE_FROM, hi: int | None = None) -> tuple:
    """Per document: the sum of the NLL change over the scored tokens [lo, hi), their count, and the count of equal
    top-1 tokens. Array index i holds token t = i + 1."""
    s, n, a = [], [], []
    for d in doc_ids:
        b_lp, x_lp = base[f"{d}_lp"].astype(np.float64), arm[f"{d}_lp"].astype(np.float64)
        top_b, top_x = base[f"{d}_top1"], arm[f"{d}_top1"]
        i0, i1 = lo - 1, (len(b_lp) if hi is None else min(hi - 1, len(b_lp)))
        if i1 <= i0:
            s.append(0.0), n.append(0), a.append(0)
            continue
        s.append(float((b_lp[i0:i1] - x_lp[i0:i1]).sum()))
        n.append(i1 - i0)
        a.append(int((top_b[i0:i1] == top_x[i0:i1]).sum()))
    return np.array(s), np.array(n, dtype=np.float64), np.array(a, dtype=np.float64)


def draws(doc_domains: list[str], boot: int = BOOT, seed: int = SEED) -> np.ndarray:
    """[boot, docs] indices of a bootstrap over documents, stratified by domain (each domain keeps its count)."""
    rng = np.random.default_rng(seed)
    cols = []
    for dom in DOMAINS:
        idx = np.array([i for i, x in enumerate(doc_domains) if x == dom])
        if len(idx):
            cols.append(idx[rng.integers(0, len(idx), size=(boot, len(idx)))])
    return np.concatenate(cols, axis=1)


def interval(s: np.ndarray, n: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """(Reported) the percentile interval of sum(s) / sum(n) over the bootstrap draws idx."""
    v = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the regularized incomplete beta function (Numerical Recipes, 2nd ed., Section 6.4)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_quantile(p: float, df: float) -> float:
    """The p quantile (p > 0.5) of Student's t with df degrees of freedom: bisection on P(T > t) = I_x(df/2, 1/2) / 2
    with x = df / (df + t^2)."""
    if not math.isfinite(df) or df > 1e7:
        return statistics.NormalDist().inv_cdf(p)
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * betainc(df / 2, 0.5, df / (df + mid * mid)) > 1 - p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t_interval(s: np.ndarray, n: np.ndarray, doms: list[str], level: float = 0.95) -> tuple[float, float]:
    """The decision interval (A1, R-77 F5): a stratified t interval of sum(s) / sum(n) over documents. The variance of
    the linearized ratio is sum_h n_h var_h(e) / N^2, with the residuals e_d = s_d - Delta n_d, var_h their sample
    variance (ddof 1) in domain h with n_h documents and N the scored tokens; the degrees of freedom are Satterthwaite's
    (sum_h u_h)^2 / sum_h u_h^2 / (n_h - 1) with u_h = n_h var_h(e)."""
    total = float(n.sum())
    est = float(s.sum()) / total
    e = s - est * n
    terms = []
    for dom in DOMAINS:
        idx = [i for i, x in enumerate(doms) if x == dom]
        if len(idx) >= 2:
            terms.append((len(idx) * float(np.var(e[idx], ddof=1)), len(idx) - 1))
    var = sum(u for u, _ in terms)
    if var <= 0.0:
        return est, est
    df = var ** 2 / sum(u ** 2 / k for u, k in terms if u > 0)
    hw = t_quantile(0.5 + level / 2, df) * math.sqrt(var) / total
    return est - hw, est + hw


def block_interval(base: dict, arm: dict, doc_ids: list[str], doc_domains: list[str], block: int = 2_048,
                   minus: dict | None = None) -> tuple[float, float]:
    """(Reported) a stratified bootstrap over 2,048-token blocks of the scored tokens (the within-document spread).
    With minus, the statistic is Delta_arm - Delta_minus (paired)."""
    sums, cnt, dom = [], [], []
    for d, x in zip(doc_ids, doc_domains):
        b = base[f"{d}_lp"].astype(np.float64)[SCORE_FROM - 1:]
        diff = b - arm[f"{d}_lp"].astype(np.float64)[SCORE_FROM - 1:]
        if minus is not None:
            diff = diff - (b - minus[f"{d}_lp"].astype(np.float64)[SCORE_FROM - 1:])
        for j in range(0, len(diff), block):
            sums.append(diff[j:j + block].sum()), cnt.append(len(diff[j:j + block])), dom.append(x)
    return interval(np.array(sums), np.array(cnt, dtype=np.float64), draws(dom, seed=SEED + 1))


def far_repeat_mask(ids, n: int = FAR_N, gap: int = FAR_GAP) -> np.ndarray:
    """(Reported, A1, R-77 O2) True at token index t if the n tokens ids[t-n+1 .. t] occurred before and their last
    earlier occurrence ended at index t - gap or before. Then the copy source lies outside the 1,024-token window of the
    query at t - 1, so only the global layers can carry it. A 64-bit rolling hash finds the n-grams."""
    a = np.asarray(ids, dtype=np.uint64)
    length = len(a)
    mask = np.zeros(length, dtype=bool)
    if length < n:
        return mask
    h = np.zeros(length - n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * np.uint64(1_000_003) + a[j:length - n + 1 + j]
    end = np.arange(n - 1, length)
    order = np.lexsort((end, h))
    hs, es = h[order], end[order]
    same = np.r_[False, hs[1:] == hs[:-1]]
    prev = np.where(same, np.r_[0, es[:-1]], -1)
    mask[es[same & (es - prev >= gap)]] = True
    return mask


def gen_metrics(gen: dict | None) -> dict:
    """GEN against BF16 (greedy, stopped at the end token or 128 tokens; A1, R-77 F10): exact matches, the first
    divergence (a shorter answer that is a prefix diverges at its end) and the token agreement."""
    if not gen:
        return {}
    base = gen["outputs"].get("BF16")
    out = {}
    for arm, outs in gen["outputs"].items():
        if arm == "BF16" or base is None:
            continue
        first, same_tok, exact = [], 0, 0
        for b, x in zip(base, outs):
            k = next((i for i, (u, v) in enumerate(zip(b, x)) if u != v), min(len(b), len(x)))
            first.append(k)
            exact += int(b == x)
            same_tok += sum(int(u == v) for u, v in zip(b, x))
        n = len(base)
        out[arm] = {"exact": exact, "prompts": n, "exact_wilson95": wilson(exact, n),
                    "first_divergence_median": statistics.median(first) if first else None,
                    "token_agreement": same_tok / max(1, sum(len(b) for b in base)),
                    "finish": {r: list(gen.get("finish", {}).get(arm, [])).count(r)
                               for r in set(gen.get("finish", {}).get(arm, []))}}
    return out


def engine_problems(arm: str, n_layers: int, ev: dict) -> tuple[list[str], list[str]]:
    """For one arm, from the hook evidence (the run record or the pre-flight): V5 (the transform checks ran and pass,
    no hook error, and the engine output equals the arm's float32 reference within GAP_TOL in every hooked layer) and
    V7 (A1, R-77 F8: in every layer where the arm's reference and the reference of the unchanged q, k and v differ by
    more than SEP_MIN, the engine output is nearer the arm's reference; BF16C decides in SEP_LAYERS_BF16C layers or
    more)."""
    v5, v7 = [], []
    if arm not in ev.get("checks", {}):
        v5.append(f"{arm}: no transform check ran")
    v5 += [f"{arm}: {b}" for b in ev.get("checks", {}).get(arm, [])]
    v5 += [f"{arm}: hook error {e}" for e in ev.get("hook_errors", {}).get(arm, [])]
    g = ev.get("gaps", {}).get(arm, {})
    if len(g) != n_layers or any(not math.isfinite(x) or x > GAP_TOL for x in g.values()):
        v5.append(f"{arm}: engine-output gaps {g} (every hooked layer at most {GAP_TOL})")
    if arm != "BF16":
        gi, se = ev.get("gaps_id", {}).get(arm, {}), ev.get("seps", {}).get(arm, {})
        decide = [lay for lay in g if se.get(lay, 0.0) > SEP_MIN]
        wrong = [lay for lay in decide if not g[lay] < gi.get(lay, math.inf)]
        if wrong:
            v7.append(f"{arm}: in layers {wrong} the engine output is nearer the unchanged reference")
        if arm == "BF16C" and len(decide) < min(SEP_LAYERS_BF16C, n_layers):
            v7.append(f"BF16C: only {len(decide)} layers have references that differ by more than {SEP_MIN}")
    return v5, v7


def preflight_problems(lps: dict, ev: dict, n_layers: int, length: int) -> list[str]:
    """The pre-flight checks on one chunk (A1, R-77 F3): V5 and V7 for every arm; full token counts; finite values;
    DET equals BF16 bit for bit or within DET_TOL; E5M2 and STOCK8 change the log-probabilities; BF16C changes them if
    DET is bit-identical (an engine that ignored the BF16C hook would give the BF16 values bit for bit)."""
    bad = [] if n_layers else ["no hooked layer"]
    for arm in HOOKED:
        v5, v7 = engine_problems(arm, n_layers, ev)
        bad += v5 + v7
        cnt = [n for key, n in ev.get("counts", {}).items() if key.startswith(f"{arm}|PRE|")]
        if len(cnt) != n_layers or any(n != length for n in cnt):
            bad.append(f"{arm}: the hooks saw {sorted(set(cnt))} tokens in {len(cnt)} layers, not {length} in "
                       f"{n_layers}")
    bad += [f"{a}: a log-probability is not finite" for a, x in lps.items() if not np.isfinite(x).all()]
    det_same = np.array_equal(lps["DET"], lps["BF16"])
    if not det_same:
        d = float(np.mean(np.asarray(lps["BF16"], np.float64) - np.asarray(lps["DET"], np.float64)))
        if abs(d) > DET_TOL:
            bad.append(f"DET differs from BF16 by {d:+.2e} nats per token (at most {DET_TOL} without bit identity)")
    bad += [f"{a} gives the BF16 log-probabilities bit for bit" for a in ("E5M2", "STOCK8")
            if np.array_equal(lps[a], lps["BF16"])]
    if det_same and np.array_equal(lps["BF16C"], lps["BF16"]):
        bad.append("BF16C gives the BF16 log-probabilities bit for bit (the engine may ignore its hook)")
    return bad


def validity(res: Path, data: dict, main: dict, arrays: dict, rehearsal: bool) -> tuple[dict, dict]:
    """V1 to V5 and V7 (V6 needs the statistics), each a list of problems, and the reported arms excluded with their
    problems (a problem of a reported arm drops the arm, not the run; A1, R-77 F2)."""
    v = {k: [] for k in V_KEYS}
    excluded: dict[str, list[str]] = {}
    stub = bool(main.get("stub"))
    gpu = str(main.get("gpu", ""))
    if not (stub or ("H100" in gpu and "80GB" in gpu)):
        v["V1"].append(f"the GPU is {gpu!r}, not an H100 80GB")
    if main.get("status") != "ok" and not main.get("decision_arms_done"):
        v["V4"].append(f"run_main status {main.get('status')} before the decision arms finished: {main.get('errors')}")
    if not rehearsal:
        if main.get("data_sha256") != DATA_SHA256:
            v["V2"].append("data_e55.json is not the frozen file")
        if main.get("scales_sha256") != SCALES_SHA256:
            v["V2"].append("scales_e54.json is not the frozen file")
        if not stub and main.get("snapshot_revision") != REVISION:
            v["V2"].append(f"the model snapshot {main.get('snapshot_revision')} is not the pinned revision")
    plan = {d["id"]: d["length"] for d in data["docs"]}
    if not rehearsal and main.get("doc_lengths") != plan:
        v["V2"].append("the documents or their lengths are not the plan")
    if not main.get("tok_check", {}).get("ok"):
        v["V2"].append("the engine tokenizer does not give the frozen ids")
    if not stub:
        e = main.get("engine", {})
        if str(e.get("kv_cache_dtype")) != "auto" or e.get("prefix_caching") not in (False, "False"):
            v["V2"].append("the cache is not BF16 auto, or prefix caching is on")
        if "NONE" not in str(e.get("compilation_mode")) or "NONE" not in str(e.get("cudagraph_mode")):
            v["V2"].append("the engine is not eager")
        if e.get("max_num_batched_tokens") != ENGINE_TOKENS and not rehearsal:
            v["V2"].append(f"{e.get('max_num_batched_tokens')} batched tokens, not {ENGINE_TOKENS}")
        want = load(res / "code_manifest.json", {})
        got = {}
        if (res / "code_sha.txt").exists():
            for line_ in (res / "code_sha.txt").read_text().split("\n"):
                if line_.strip():
                    h, name = line_.split()[:2]
                    got[Path(name).name] = h
        for name in CODE_FILES:
            if not want.get(name) or got.get(name) != want.get(name):
                v["V2"].append(f"code hash of {name}: Pod {got.get(name)} against design {want.get(name)} (L-25)")
    hooked = main.get("hooks", {}).get("hooked", [])
    layers = [h["layer"] for h in hooked]
    if not rehearsal:
        if layers != list(GLOBAL_LAYERS):
            v["V3"].append(f"hooked layers {layers}, not the 10 global layers")
        if not stub and any((h["num_heads"], h["num_kv_heads"], h["head_size"]) != HEADS for h in hooked):
            v["V3"].append("a hooked layer has other head counts or head size")
    v["V3"] += main.get("hooks", {}).get("problems", [])
    counts = main.get("counts", {})
    for arm in HOOKED:
        if arm not in main.get("arms", {}):
            continue
        p3 = []
        for d in main["docs"]:
            bad = [lay for lay in layers if counts.get(f"{arm}|{d}|{lay}") != main["doc_lengths"][d]]
            if bad:
                p3.append(f"{arm} {d}: in layers {bad} the hook did not see {main['doc_lengths'][d]} tokens")
        p5, p7 = engine_problems(arm, len(layers), main)
        if arm in DECISION:
            v["V3"] += p3
            v["V5"] += p5
            v["V7"] += p7
        elif p3 or p5 or p7:
            excluded[arm] = p3 + p5 + p7
    if "DET" in arrays and "BF16" in arrays:
        det = main["arms"].get("DET", {}).get("docs", [])
        same = all(np.array_equal(arrays["DET"][f"{d}_{f}"], arrays["BF16"][f"{d}_{f}"])
                   for d in det for f in ("lp", "top1"))
        if not same:   # the rerun is not bit-identical: it must stay far inside the margin of H16
            s, n, _ = doc_sums(arrays["BF16"], arrays["DET"], det)
            if not det or abs(s.sum() / n.sum()) > DET_TOL:
                v["V3"].append(f"DET differs from BF16 by {s.sum() / max(n.sum(), 1):+.2e} nats per token "
                               f"(at most {DET_TOL} without bit identity)")
    else:
        v["V3"].append("no determinism rerun")
    for arm in DECISION:
        if arm not in arrays:
            v["V4"].append(f"the decision arm {arm} is missing")
    for arm, a in arrays.items():
        probs = []
        for d in main.get("docs", []):
            lp = a.get(f"{d}_lp")
            if lp is None:
                if arm != "DET":
                    probs.append(f"{arm} {d}: no array")
                continue
            if len(lp) != main["doc_lengths"][d] - 1 or not np.isfinite(lp).all():
                probs.append(f"{arm} {d}: {len(lp)} values or a value that is not finite")
        if probs and (arm in DECISION or arm == "DET"):
            v["V4"] += probs
        elif probs:
            excluded.setdefault(arm, []).extend(probs)
    return v, excluded


def decide16(st: dict) -> tuple[str, str]:
    x, (lo, hi) = st["delta"]["BF16C"], st["ci"]["BF16C"]
    m = st["m16"]
    if x > KILL16:
        return "KILL16", f"Delta_BF16C {x:.5f} > {KILL16}"
    if -m <= lo and hi <= m:
        return "GO16", f"95% interval [{lo:.6f}, {hi:.6f}] inside +-{m:.6f}"
    if lo > m:
        return "PIVOT16", f"95% interval lower bound {lo:.6f} > margin {m:.6f}, Delta {x:.5f} <= {KILL16}"
    if hi < -m:
        return "CHECK16", f"95% interval upper bound {hi:.6f} < -{m:.6f}: an improvement that rounding cannot give"
    return "INCONCLUSIVE16", f"95% interval [{lo:.6f}, {hi:.6f}] not inside +-{m:.6f}"


def decide8(st: dict) -> tuple[str, str]:
    d, (lo, hi), tol = st["D"], st["ci_D"], st["tol8"]
    e, (elo, ehi) = st["E"], st["ci_E"]
    if hi <= tol:
        return "GO8", f"D {d:.6f}, 95% upper bound {hi:.6f} <= tolerance {tol:.6f}"
    if elo > 0:
        return "KILL8", (f"E = Delta_TIED8 - {KILL8_RATIO} Delta_STOCK8 {e:.6f}, 95% interval [{elo:.6f}, {ehi:.6f}] "
                         f"> 0: TIED8 is worse than {KILL8_RATIO} x STOCK8")
    if lo > 0:
        return "PIVOT8", f"D {d:.6f} [{lo:.6f}, {hi:.6f}] > 0; E {e:.6f} [{elo:.6f}, {ehi:.6f}] not above 0"
    return "INCONCLUSIVE8", f"D {d:.6f} [{lo:.6f}, {hi:.6f}]: upper bound above {tol:.6f}"


def summarize(res: Path, data_path: Path) -> dict:
    data = json.loads(data_path.read_text())
    main = load(res / "run_main.json", {})
    rehearsal = bool(main.get("rehearsal"))
    arrays = {p.stem: dict(np.load(p)) for p in sorted((res / "arrays").glob("*.npz"))}
    v, excluded = validity(res, data, main, arrays, rehearsal)
    for arm in excluded:
        arrays.pop(arm, None)
    out = {"validity": v, "excluded": excluded, "rehearsal": rehearsal, "status": main.get("status"),
           "cut": main.get("cut", []), "step_errors": main.get("step_errors", []),
           "preflight": (main.get("preflight") or {}).get("problems")}
    if any(v[k] for k in V_KEYS) and not rehearsal:
        out["decision"] = "INVALID"
        out["failed"] = [k for k in V_KEYS if v[k]]
        return out
    if any(a not in arrays for a in DECISION):
        out["decision"] = "INVALID"
        out["failed"] = ["V4"]
        return out
    ids = list(main["docs"])
    doms = [next(d["domain"] for d in data["docs"] if d["id"] == x) for x in ids]
    plan_ids = {d["id"]: d["ids"] for d in data["docs"]}
    idx = draws(doms)
    base = arrays["BF16"]
    st = {"delta": {}, "ci": {}, "ci_boot": {}, "agree": {}, "per_domain": {}, "per_bin": {}, "all_tokens": {},
          "far": {}, "sigma": {}, "scored": None}
    far = {}
    for d in ids:
        length = main["doc_lengths"][d]
        far[d] = far_repeat_mask(plan_ids[d][:length])[1:length] & (np.arange(length - 1) >= SCORE_FROM - 1)
    nf = np.array([far[d].sum() for d in ids], dtype=np.float64)
    S = {}
    for arm in SCORED:
        if arm not in arrays:
            continue
        s, n, a = doc_sums(base, arrays[arm], ids)
        S[arm] = s
        st["scored"] = int(n.sum())
        st["delta"][arm] = float(s.sum() / n.sum())
        st["ci"][arm] = t_interval(s, n, doms)
        st["ci_boot"][arm] = interval(s, n, idx)
        st["agree"][arm] = float(a.sum() / n.sum())
        st["per_domain"][arm] = {dom: float(s[[i for i, x in enumerate(doms) if x == dom]].sum()
                                            / n[[i for i, x in enumerate(doms) if x == dom]].sum()) for dom in DOMAINS}
        st["per_bin"][arm] = {}
        for lo, hi in BINS:
            sb, nb, ab = doc_sums(base, arrays[arm], ids, lo, hi)
            if nb.sum():
                st["per_bin"][arm][f"{lo}-{hi}"] = {"delta": float(sb.sum() / nb.sum()),
                                                     "agree": float(ab.sum() / nb.sum()), "tokens": int(nb.sum())}
        s1, n1, _ = doc_sums(base, arrays[arm], ids, 1)          # all tokens: index 1 or more (A1, R-77 F9)
        st["all_tokens"][arm] = float(s1.sum() / n1.sum())
        diffs = {d: base[f"{d}_lp"].astype(np.float64) - arrays[arm][f"{d}_lp"].astype(np.float64) for d in ids}
        sf = np.array([diffs[d][far[d]].sum() for d in ids])
        st["far"][arm] = {"tokens": int(nf.sum()), "delta": float(sf.sum() / nf.sum()) if nf.sum() else None,
                          "ci": t_interval(sf, nf, doms) if nf.sum() else None}
        st["sigma"][arm] = float(np.concatenate([diffs[d][SCORE_FROM - 1:] for d in ids]).std())
    det = main["arms"].get("DET", {}).get("docs", [])
    if "DET" in arrays and det:
        sd, nd, ad = doc_sums(base, arrays["DET"], det)
        st["det"] = {"docs": det, "delta": float(sd.sum() / nd.sum()), "agree": float(ad.sum() / nd.sum()),
                     "bit_identical": all(np.array_equal(arrays["DET"][f"{d}_{f}"], base[f"{d}_{f}"])
                                          for d in det for f in ("lp", "top1"))}
    n = doc_sums(base, arrays["STOCK8"], ids)[1]
    st["D"] = st["delta"]["TIED8"] - st["delta"]["STOCK8"]
    st["ci_D"] = t_interval(S["TIED8"] - S["STOCK8"], n, doms)
    st["ci_D_boot"] = interval(S["TIED8"] - S["STOCK8"], n, idx)
    st["E"] = st["delta"]["TIED8"] - KILL8_RATIO * st["delta"]["STOCK8"]
    st["ci_E"] = t_interval(S["TIED8"] - KILL8_RATIO * S["STOCK8"], n, doms)
    st["m16"] = M16
    st["m16_rel"] = M16_REL * st["delta"]["E5M2"]
    st["inside_m16_rel"] = bool(-st["m16_rel"] <= st["ci"]["BF16C"][0] and st["ci"]["BF16C"][1] <= st["m16_rel"])
    st["tol8"] = max(TOL8_ABS, TOL8_REL * st["delta"]["STOCK8"])
    st["tol8_pct_of_stock8"] = 100 * st["tol8"] / st["delta"]["STOCK8"] if st["delta"]["STOCK8"] > 0 else None
    st["M1"] = st["delta"]["TIED8"] / st["delta"]["STOCK8"] if st["delta"]["STOCK8"] else math.nan
    st["M2"] = st["delta"]["TIED8"] / st["delta"]["JOINT8"] if st["delta"].get("JOINT8") else None
    if "BF16R" in S:
        st["bf16c_minus_bf16r"] = {"delta": st["delta"]["BF16C"] - st["delta"]["BF16R"],
                                   "ci": t_interval(S["BF16C"] - S["BF16R"], n, doms)}
    st["block_ci"] = {"BF16C": block_interval(base, arrays["BF16C"], ids, doms),
                      "D": block_interval(base, arrays["TIED8"], ids, doms, minus=arrays["STOCK8"])}
    st["gen"] = gen_metrics(load(res / "gen_outputs.json"))
    out["stats"] = st
    v["V6"] = [] if st["ci"]["E5M2"][0] > 0 else [f"E5M2 95% interval {st['ci']['E5M2']} does not exclude 0"]
    if v["V6"] and not rehearsal:
        out["decision"] = "INVALID"
        out["failed"] = ["V6"]
        return out
    h16, why16 = decide16(st)
    h8, why8 = decide8(st)
    out.update({"decision": f"{h16} {h8}", "H16": h16, "H16_reason": why16, "H8": h8, "H8_reason": why8,
                "action": {h16: ACTION[h16], h8: ACTION[h8]}})
    return out


def line(s: dict) -> str:
    probs = "; ".join(f"{k}: {p[:2]}" for k, p in s["validity"].items() if p)
    if s["decision"] == "INVALID":
        pre = f"; pre-flight: {s['preflight'][:3]}" if s.get("preflight") else ""
        return f"INVALID ({', '.join(s.get('failed', []))}): {probs}{pre}"[:900]
    st = s["stats"]
    d, ci = st["delta"], st["ci"]
    fmt = lambda a: f"{d[a]:+.5f} [{ci[a][0]:+.5f}, {ci[a][1]:+.5f}]" if a in d else "not run"  # noqa: E731
    text = (f"{'REHEARSAL ' if s.get('rehearsal') else ''}{s['H16']} {s['H8']}: Delta_BF16C {fmt('BF16C')} (margin "
            f"{st['m16']:.5f}; 0.1 x E5M2 {st['m16_rel']:.5f}); D {st['D']:+.5f} [{st['ci_D'][0]:+.5f}, "
            f"{st['ci_D'][1]:+.5f}] (tolerance {st['tol8']:.5f}); E {st['E']:+.5f} [{st['ci_E'][0]:+.5f}, "
            f"{st['ci_E'][1]:+.5f}]; STOCK8 {fmt('STOCK8')}; TIED8 {fmt('TIED8')}; E5M2 {fmt('E5M2')}; BF16R "
            f"{fmt('BF16R')}; JOINT8 {fmt('JOINT8')}; TIED8_UP {fmt('TIED8_UP')}; top-1 agreement BF16C "
            f"{st['agree'].get('BF16C', math.nan):.4f}, BF16R {st['agree'].get('BF16R', math.nan):.4f}; cut "
            f"{[c['step'] for c in s['cut']]}")
    if s.get("excluded"):
        text += f"; excluded {sorted(s['excluded'])}"
    if s.get("step_errors"):
        text += f"; step errors {[e['step'] for e in s['step_errors']]}"
    if probs:                                            # shown in a rehearsal (R-77 B2)
        text += f"; validity problems: {probs}"[:600]
    return text


def manifest(s: dict, res: Path, design_commit: str) -> dict:
    """The ledger manifest of the run (MEASUREMENT.md Section 5): one value per metric (no repeats)."""
    main = load(res / "run_main.json", {})
    st = s.get("stats", {})
    metrics = {f"dnll_{a}": {"values": [v], "unit": "nats/token", "higher_is_better": False}
               for a, v in st.get("delta", {}).items()}
    metrics.update({f"top1_agree_{a}": {"values": [v], "unit": "fraction", "higher_is_better": True}
                    for a, v in st.get("agree", {}).items()})
    return {"run_id": res.name, "experiment": "E-55", "idea": "I-182",
            "status": "ok" if s.get("decision") != "INVALID" else "failed", "design_commit": design_commit,
            "config": main.get("engine_kwargs", {}), "started_at": main.get("started_at"),
            "finished_at": main.get("finished_at"),
            "hardware": {"gpu_name": main.get("gpu"), "gpu_count": 1, "driver": None, "cuda": None},
            "software": {"vllm": main.get("engine", {}).get("vllm")}, "metrics": metrics,
            "results_path": str(res), "notes": f"decision={s['decision']}; {line(s)}"[:2000]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=HERE / "data_e55.json")
    ap.add_argument("--manifest-out", type=Path)
    ap.add_argument("--design-commit", default="")
    args = ap.parse_args()
    s = summarize(args.res, args.data)
    (args.res / "summary.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    if args.manifest_out:
        args.manifest_out.write_text(json.dumps(manifest(s, args.res, args.design_commit), indent=1) + "\n")
    print(line(s))


if __name__ == "__main__":
    main()

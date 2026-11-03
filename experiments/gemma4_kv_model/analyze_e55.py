"""E-55 analysis (README Sections 5 to 7): the statistics, the validity checks and the frozen decision rule, from the
files of one run (`run_main.json`, `run_noise.json`, `arrays/*.npz`, `gen_outputs.json`, `code_sha.txt`).

Statistic (README Section 5): for an arm X and a document d, S_X(d) is the sum over the scored tokens (index t >= 1025:
the context holds more than the 1,024-token sliding window) of the BF16 log-probability minus the arm's log-probability
(the NLL change), and n(d) the number of scored tokens. Delta_X = sum_d S_X(d) / sum_d n(d), in nats per token. The 95%
interval comes from a paired bootstrap over documents, stratified by domain (4 books, 4 prose, 4 code), 10,000 draws,
seed 55. D = Delta_TIED8 - Delta_STOCK8 uses the same draws.

Usage: analyze_e55.py --res <dir> [--data data_e55.json] [--manifest-out <file> --design-commit <sha>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_SHA256 = "8f2e2ecd96402b51c13d2f73f25cfe6e0a10c7b6c07a97d0925a334441df36e4"   # data_e55.json (V2; frozen)
SCALES_SHA256 = "8efe6e9076e28e011ff560106665998434b9403a1ddd9f653bdd98e5c718bd43"   # scales_e54.json (E-54 A1)
REVISION = "d4ab4f579dd3516f97d8a6a4c98d0653480bad15"
GLOBAL_LAYERS = (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
HEADS = (32, 4, 512)
DOMAINS = ("books", "prose", "code")
SCORED = ("BF16C", "E5M2", "STOCK8", "TIED8", "JOINT8", "TIED8_UP", "BF16_B")
DECISION = ("BF16", "BF16C", "E5M2", "STOCK8", "TIED8")
SCORE_FROM = 1_025                      # the first scored token index (context of 1,025 tokens or more)
BINS = ((1_025, 8_192), (8_192, 32_768), (32_768, 65_536))
BOOT, SEED = 10_000, 55
M16_ABS, M16_REL, KILL16 = 0.001, 0.1, 0.01          # H16: margin min(0.001, 0.1 Delta_E5M2); KILL above 0.01
TOL8_ABS, TOL8_REL, KILL8_RATIO = 0.0002, 0.25, 1.5  # H8: tolerance max(0.0002, 0.25 Delta_STOCK8)
ENGINE_TOKENS = {"main": 4_096, "noise": 2_048}
GAP_TOL = 0.01                          # V5: the engine output against the float32 reference of the arm (E-54 0.0017)
DET_TOL = 1e-4                          # V3: a rerun that is not bit-identical may differ by this much (M16_ABS / 10)
CODE_FILES = ("splice_e55.py", "run_e55.py", "analyze_e55.py", "session_e55.sh", "data_e55.json", "scales_e54.json")
ACTION = {
    "GO16": "the thesis and the paper claim that the compact BF16 cache keeps the model's next-token predictions "
            "(model level, 3 domains, 32K to 64K tokens); the BF16 compact build (D-88) can start, then the engine "
            "capacity test",
    "PIVOT16": "the compact BF16 cache has a small measured cost: report it; the user decides on the build",
    "KILL16": "stop the 16-bit claim and find the cause (a fault is likely, because the layout is exact in real "
              "arithmetic) before any new run",
    "INCONCLUSIVE16": "report; the user decides on more documents or another metric (a new S4)",
    "GO8": "the compact FP8 row stays a build option after the BF16 build",
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
    v = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


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


def gen_metrics(gen: dict | None) -> dict:
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
                    "token_agreement": same_tok / max(1, sum(len(b) for b in base))}
    return out


def validity(res: Path, data: dict, main: dict, noise: dict | None, arrays: dict, rehearsal: bool) -> dict:
    """V1 to V5 (V6 needs the statistics). Each value is a list of problems."""
    v = {f"V{i}": [] for i in range(1, 6)}
    gpu = str(main.get("gpu", ""))
    if not (main.get("stub") or ("H100" in gpu and "80GB" in gpu)):
        v["V1"].append(f"the GPU is {gpu!r}, not an H100 80GB")
    if main.get("status") != "ok":
        v["V4"].append(f"run_main status {main.get('status')}: {main.get('errors')}")
    if not rehearsal:
        if main.get("data_sha256") != DATA_SHA256:
            v["V2"].append("data_e55.json is not the frozen file")
        if main.get("scales_sha256") != SCALES_SHA256:
            v["V2"].append("scales_e54.json is not the frozen file")
        if main.get("revision") != REVISION:
            v["V2"].append("the model revision is not the pinned one")
    plan = {d["id"]: d["length"] for d in data["docs"]}
    if not rehearsal and main.get("doc_lengths") != plan:
        v["V2"].append("the documents or their lengths are not the plan")
    if not main.get("tok_check", {}).get("ok"):
        v["V2"].append("the engine tokenizer does not give the frozen ids")
    for phase, rec in (("main", main), ("noise", noise)):
        if rec is None or rec.get("stub"):
            continue
        e = rec.get("engine", {})
        if str(e.get("kv_cache_dtype")) != "auto" or e.get("prefix_caching") not in (False, "False"):
            v["V2"].append(f"{phase}: the cache is not BF16 auto, or prefix caching is on")
        if "NONE" not in str(e.get("compilation_mode")) or "NONE" not in str(e.get("cudagraph_mode")):
            v["V2"].append(f"{phase}: the engine is not eager")
        if e.get("max_num_batched_tokens") != ENGINE_TOKENS[phase] and not rehearsal:
            v["V2"].append(f"{phase}: {e.get('max_num_batched_tokens')} batched tokens, not {ENGINE_TOKENS[phase]}")
    want = load(res / "code_manifest.json", {})
    got = {}
    if (res / "code_sha.txt").exists():
        for line in (res / "code_sha.txt").read_text().split("\n"):
            if line.strip():
                h, name = line.split()[:2]
                got[Path(name).name] = h
    if not main.get("stub"):
        for name in CODE_FILES:
            if not want.get(name) or got.get(name) != want.get(name):
                v["V2"].append(f"code hash of {name}: Pod {got.get(name)} against design {want.get(name)} (L-25)")
    hooked = main.get("hooks", {}).get("hooked", [])
    if not rehearsal:
        if [h["layer"] for h in hooked] != list(GLOBAL_LAYERS):
            v["V3"].append(f"hooked layers {[h['layer'] for h in hooked]}, not the 10 global layers")
        if any((h["num_heads"], h["num_kv_heads"], h["head_size"]) != HEADS for h in hooked):
            v["V3"].append("a hooked layer has other head counts or head size")
    v["V3"] += main.get("hooks", {}).get("problems", [])
    counts = main.get("counts", {})
    layers = [h["layer"] for h in hooked]
    for arm in ("BF16", "BF16C", "E5M2", "STOCK8", "TIED8", "JOINT8", "TIED8_UP"):
        if arm not in main.get("arms", {}):
            continue
        for d in main["docs"]:
            for layer in layers:
                if counts.get(f"{arm}|{d}|{layer}") != main["doc_lengths"][d]:
                    v["V3"].append(f"{arm} {d} layer {layer}: the hook saw {counts.get(f'{arm}|{d}|{layer}')} "
                                   f"tokens, not {main['doc_lengths'][d]}")
                    break
    if "DET" in arrays and "BF16" in arrays:
        det = main["arms"].get("DET", {}).get("docs", [])
        same = all(np.array_equal(arrays["DET"][f"{d}_{f}"], arrays["BF16"][f"{d}_{f}"])
                   for d in det for f in ("lp", "top1"))
        if not same:   # the rerun is not bit-identical: it must stay far inside the smallest margin of H16
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
        for d in main["docs"]:
            lp = a.get(f"{d}_lp")
            if lp is None:
                if arm != "DET":
                    v["V4"].append(f"{arm} {d}: no array")
                continue
            if len(lp) != main["doc_lengths"][d] - 1 or not np.isfinite(lp).all():
                v["V4"].append(f"{arm} {d}: {len(lp)} values or a value that is not finite")
    for arm, bad in main.get("checks", {}).items():
        v["V5"] += [f"{arm}: {b}" for b in bad]
    gaps = main.get("gaps", {})
    for arm in ("BF16", "BF16C", "E5M2", "STOCK8", "TIED8", "JOINT8", "TIED8_UP"):
        if arm not in main.get("arms", {}):
            continue
        g = gaps.get(arm, {})
        if len(g) != len(layers) or any(not math.isfinite(x) or x > GAP_TOL for x in g.values()):
            v["V5"].append(f"{arm}: engine-output gaps {g} (every hooked layer at most {GAP_TOL})")
    for arm in DECISION:
        if arm in arrays and arm not in main.get("checks", {}) and not main.get("stub"):
            v["V5"].append(f"{arm}: no transform check ran")
    v["V5"] += main.get("hook_errors", [])
    return v


def decide16(st: dict) -> tuple[str, str]:
    x, (lo, hi) = st["delta"]["BF16C"], st["ci"]["BF16C"]
    m = st["m16"]
    if x > KILL16:
        return "KILL16", f"Delta_BF16C {x:.5f} > {KILL16}"
    if -m <= lo and hi <= m:
        return "GO16", f"95% interval [{lo:.6f}, {hi:.6f}] inside +-{m:.6f}"
    if lo > m:
        return "PIVOT16", f"95% interval lower bound {lo:.6f} > margin {m:.6f}, Delta {x:.5f} <= {KILL16}"
    return "INCONCLUSIVE16", f"95% interval [{lo:.6f}, {hi:.6f}] not inside +-{m:.6f}"


def decide8(st: dict) -> tuple[str, str]:
    d, (lo, hi), tol = st["D"], st["ci_D"], st["tol8"]
    t, s = st["delta"]["TIED8"], st["delta"]["STOCK8"]
    if hi <= tol:
        return "GO8", f"D {d:.6f}, 95% upper bound {hi:.6f} <= tolerance {tol:.6f}"
    if lo > 0 and t > KILL8_RATIO * s:
        return "KILL8", f"D {d:.6f} [{lo:.6f}, {hi:.6f}] > 0 and Delta_TIED8 {t:.6f} > {KILL8_RATIO} x {s:.6f}"
    if lo > 0:
        return "PIVOT8", f"D {d:.6f} [{lo:.6f}, {hi:.6f}] > 0, within {KILL8_RATIO} x Delta_STOCK8"
    return "INCONCLUSIVE8", f"D {d:.6f} [{lo:.6f}, {hi:.6f}]: upper bound above {tol:.6f}"


def summarize(res: Path, data_path: Path) -> dict:
    data = json.loads(data_path.read_text())
    main = load(res / "run_main.json", {})
    noise = load(res / "run_noise.json")
    rehearsal = bool(main.get("rehearsal"))
    arrays = {p.stem: dict(np.load(p)) for p in sorted((res / "arrays").glob("*.npz"))}
    v = validity(res, data, main, noise, arrays, rehearsal)
    out = {"validity": v, "rehearsal": rehearsal, "cut": main.get("cut", []) + (noise or {}).get("cut", [])}
    if any(v[k] for k in ("V1", "V2", "V3", "V4", "V5")) and not rehearsal:
        out["decision"] = "INVALID"
        out["failed"] = [k for k in ("V1", "V2", "V3", "V4", "V5") if v[k]]
        return out
    if any(a not in arrays for a in DECISION):
        out["decision"] = "INVALID"
        out["failed"] = ["V4"]
        return out
    ids = list(main["docs"])
    doms = [next(d["domain"] for d in data["docs"] if d["id"] == x) for x in ids]
    idx = draws(doms)
    base = arrays["BF16"]
    st = {"delta": {}, "ci": {}, "agree": {}, "per_domain": {}, "per_bin": {}, "all_tokens": {}, "scored": None}
    S = {}
    for arm in SCORED:
        if arm not in arrays:
            continue
        s, n, a = doc_sums(base, arrays[arm], ids)
        S[arm] = s
        st["scored"] = int(n.sum())
        st["delta"][arm] = float(s.sum() / n.sum())
        st["ci"][arm] = interval(s, n, idx)
        st["agree"][arm] = float(a.sum() / n.sum())
        st["per_domain"][arm] = {dom: float(s[[i for i, x in enumerate(doms) if x == dom]].sum()
                                            / n[[i for i, x in enumerate(doms) if x == dom]].sum()) for dom in DOMAINS}
        st["per_bin"][arm] = {}
        for lo, hi in BINS:
            sb, nb, ab = doc_sums(base, arrays[arm], ids, lo, hi)
            if nb.sum():
                st["per_bin"][arm][f"{lo}-{hi}"] = {"delta": float(sb.sum() / nb.sum()),
                                                     "agree": float(ab.sum() / nb.sum()), "tokens": int(nb.sum())}
        s1, n1, _ = doc_sums(base, arrays[arm], ids, 2)
        st["all_tokens"][arm] = float(s1.sum() / n1.sum())
    det = main["arms"].get("DET", {}).get("docs", [])
    if "DET" in arrays and det:
        sd, nd, ad = doc_sums(base, arrays["DET"], det)
        st["det"] = {"docs": det, "delta": float(sd.sum() / nd.sum()), "agree": float(ad.sum() / nd.sum()),
                     "bit_identical": all(np.array_equal(arrays["DET"][f"{d}_{f}"], base[f"{d}_{f}"])
                                          for d in det for f in ("lp", "top1"))}
    n = doc_sums(base, arrays["STOCK8"], ids)[1]
    st["D"] = st["delta"]["TIED8"] - st["delta"]["STOCK8"]
    st["ci_D"] = interval(S["TIED8"] - S["STOCK8"], n, idx)
    st["m16"] = min(M16_ABS, M16_REL * st["delta"]["E5M2"])
    st["tol8"] = max(TOL8_ABS, TOL8_REL * st["delta"]["STOCK8"])
    st["M1"] = st["delta"]["TIED8"] / st["delta"]["STOCK8"] if st["delta"]["STOCK8"] else math.nan
    st["M2"] = (st["delta"]["TIED8"] / st["delta"]["JOINT8"]
                if st["delta"].get("JOINT8") else None)
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
    if s["decision"] == "INVALID":
        return f"INVALID ({', '.join(s.get('failed', []))}): " + "; ".join(
            f"{k}: {p[:2]}" for k, p in s["validity"].items() if p)[:600]
    st = s["stats"]
    d, ci = st["delta"], st["ci"]
    fmt = lambda a: f"{d[a]:+.5f} [{ci[a][0]:+.5f}, {ci[a][1]:+.5f}]" if a in d else "not run"  # noqa: E731
    pre = "REHEARSAL " if s.get("rehearsal") else ""
    return (f"{pre}{s['H16']} {s['H8']}: Delta_BF16C {fmt('BF16C')} (margin {st['m16']:.5f}); D {st['D']:+.5f} "
            f"[{st['ci_D'][0]:+.5f}, {st['ci_D'][1]:+.5f}] (tolerance {st['tol8']:.5f}); STOCK8 {fmt('STOCK8')}; "
            f"TIED8 {fmt('TIED8')}; JOINT8 {fmt('JOINT8')}; E5M2 {fmt('E5M2')}; TIED8_UP {fmt('TIED8_UP')}; "
            f"BF16_B {fmt('BF16_B')}; top-1 agreement BF16C {st['agree'].get('BF16C', math.nan):.4f}, BF16_B "
            f"{st['agree'].get('BF16_B', math.nan):.4f}; cut {[c['step'] for c in s['cut']]}")


def manifest(s: dict, res: Path, design_commit: str) -> dict:
    """The ledger manifest of the run (MEASUREMENT.md Section 5): one value per metric (no repeats)."""
    main = load(res / "run_main.json", {})
    st = s.get("stats", {})
    metrics = {f"dnll_{a}": {"values": [v], "unit": "nats/token", "higher_is_better": False}
               for a, v in st.get("delta", {}).items()}
    metrics.update({f"top1_agree_{a}": {"values": [v], "unit": "fraction", "higher_is_better": True}
                    for a, v in st.get("agree", {}).items()})
    return {"run_id": res.name, "experiment": "E-55", "idea": "I-182", "status": "ok" if main.get("status") == "ok"
            else "failed", "design_commit": design_commit, "config": main.get("engine_kwargs", {}),
            "started_at": main.get("started_at"), "finished_at": main.get("finished_at"),
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

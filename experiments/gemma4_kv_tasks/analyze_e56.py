"""E-56 analysis (README Sections 5 to 7): the grading of the task answers, the paired statistics, the validity checks
V1 to V5, V7 and V8 (the baseline health, L-29, R-78) and the frozen decision rule, from the files of one run
(`run_main.json`, `outputs_<arm>.json`, `arrays/<arm>.npz`, `code_sha.txt`, `code_manifest.json`).
`preflight_problems` and `health_problems` are also the run-time checks of `run_e56.py`.

Primary (H1): the task accuracy of the compact BF16 cache (BF16C) minus that of the BF16 cache, over the 432 items
(400 GSM8K, 32 needle), with a paired 95% interval (the Agresti-Min adjusted Wald interval: 0.5 added to each cell of
the 2 x 2 table). Secondary: H2, the share of items whose final answer differs from the BF16 answer, BF16C minus BF16R
(the rounding-size noise arm), with the same interval; H3, the change of the mean NLL of the GSM8K reference answers
(nats per token, Delta = sum of the per-item NLL changes / sum of the answer tokens), with a t interval over items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E55 = HERE.parent / "gemma4_kv_model"
sys.path.append(str(E55))                    # after this folder: E-55 has files of the same names
import analyze_e55 as an55  # noqa: E402

DATA_SHA256 = "4f8511c89b77481ee5f2b39927de41c99b6da63680ccd8c5f6bcccd6431db672"   # data_e56.json (V2; frozen)
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"                               # google/gemma-4-31B-it
GLOBAL_LAYERS = an55.GLOBAL_LAYERS
ARMS = ("BF16", "BF16C", "BF16R", "STOCK8")          # the hooked arms of E-56 (a subset of splice_e55.ARMS)
DECISION = ("BF16", "DET", "BF16C", "BF16R")         # must finish (V4); STOCK8 is reported
ENGINE_TOKENS = 2_048
GAP_TOL, DET_TOL = an55.GAP_TOL, an55.DET_TOL
GSM_MIN, NEEDLE_MIN, REF_NLL_MAX = 0.75, 0.75, 1.5   # V8: the baseline health of BF16 (project heuristics)
M_ACC, M_DIS, M_NLL, KILL_NLL = 0.03, 0.03, 0.002, 0.01
V_KEYS = ("V1", "V2", "V3", "V4", "V5", "V7", "V8")
CODE_FILES = ("splice_e55.py", "run_e55.py", "analyze_e55.py", "run_e56.py", "analyze_e56.py", "session_e56.sh",
              "data_e56.json")
NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
PHRASE = re.compile(r"[a-z]+-\d{4}")


def load(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def extract_gsm(text: str) -> tuple[str | None, str]:
    """The final number of a GSM8K answer: the first number after the last '####', else the last number in the text.
    Returns (the number as text without commas, or None; the method)."""
    if "####" in text:
        m = NUM.search(text.rsplit("####", 1)[1])
        if m:
            return m.group(0).replace(",", "").replace("$", ""), "####"
    found = NUM.findall(text)
    return (found[-1].replace(",", "").replace("$", ""), "last") if found else (None, "none")


def gsm_correct(answer: str | None, gold: str) -> bool:
    a, g = (_num(answer) if answer is not None else None), _num(gold)
    return a is not None and g is not None and abs(a - g) < 1e-6


def extract_needle(text: str) -> str:
    """The passphrase in a needle answer (word-NNNN), else the normalized text."""
    t = text.strip().lower()
    m = PHRASE.search(t)
    return m.group(0) if m else re.sub(r"\s+", " ", t).strip(" .")


def grade(data: dict, outputs: dict) -> dict:
    """item id -> {"answer", "correct", "ids"} for the GSM8K and needle answers of one arm."""
    out = {}
    for g in data["gsm8k"]:
        o = outputs.get("gsm8k", {}).get(g["id"])
        if o is None:
            continue
        ans, how = extract_gsm(o["text"])
        out[g["id"]] = {"answer": ans, "how": how, "correct": gsm_correct(ans, g["gold"]), "ids": o["ids"]}
    for n in data["needle"]:
        o = outputs.get("needle", {}).get(n["id"])
        if o is None:
            continue
        ans = extract_needle(o["text"])
        out[n["id"]] = {"answer": ans, "how": "phrase", "correct": n["gold"].lower() in o["text"].lower(),
                        "ids": o["ids"]}
    return out


def paired_diff(x: list[bool], y: list[bool], z: float = 1.959964) -> tuple[float, tuple[float, float]]:
    """mean(y) - mean(x) for paired binary outcomes, with the Agresti-Min interval (0.5 added to each cell)."""
    n = len(x)
    b = sum(1 for a, c in zip(x, y) if a and not c)        # x yes, y no
    c = sum(1 for a, cc in zip(x, y) if not a and cc)      # x no, y yes
    d = (c - b) / n if n else float("nan")
    b2, c2, n2 = b + 0.5, c + 0.5, n + 2
    p12, p21 = b2 / n2, c2 / n2
    dd = p21 - p12
    se = math.sqrt(max((p12 + p21 - dd * dd) / n2, 0.0))
    return d, (dd - z * se, dd + z * se)


def ratio_interval(s: np.ndarray, n: np.ndarray, level: float = 0.95) -> tuple[float, tuple[float, float]]:
    """Delta = sum(s) / sum(n) over items, with a t interval of the linearized ratio (df = items - 1)."""
    k = len(s)
    delta = float(s.sum() / n.sum())
    e = s - delta * n
    se = math.sqrt(float(np.var(e, ddof=1)) * k) / float(n.sum()) if k > 1 else float("nan")
    t = an55.t_quantile(0.5 + level / 2, k - 1) if k > 1 else float("nan")
    return delta, (delta - t * se, delta + t * se)


def ref_nll(arr: dict, data: dict) -> dict:
    """item id -> (the sum of the reference-answer log-probabilities, the number of answer tokens)."""
    out = {}
    for g in data["gsm8k"]:
        lp = arr.get(f"{g['id']}_lp")
        if lp is not None:
            out[g["id"]] = (float(np.asarray(lp, np.float64).sum()), int(len(lp)))
    return out


def health_problems(graded: dict, ref: dict, data: dict) -> tuple[list[str], dict]:
    """V8 (L-29, R-78): the BF16 baseline answers and scores well enough for a comparison to mean something."""
    gsm = [graded[g["id"]]["correct"] for g in data["gsm8k"] if g["id"] in graded]
    ndl = [graded[n["id"]]["correct"] for n in data["needle"] if n["id"] in graded]
    tot = sum(v[0] for v in ref.values())
    cnt = sum(v[1] for v in ref.values())
    m = {"gsm8k_acc": float(np.mean(gsm)) if gsm else float("nan"),
         "needle_acc": float(np.mean(ndl)) if ndl else float("nan"),
         "ref_nll": -tot / cnt if cnt else float("nan"), "gsm8k_n": len(gsm), "needle_n": len(ndl)}
    bad = []
    if len(gsm) != len(data["gsm8k"]) or len(ndl) != len(data["needle"]):
        bad.append(f"BF16 answers: {len(gsm)} GSM8K and {len(ndl)} needle, not all items")
    if not m["gsm8k_acc"] >= GSM_MIN:
        bad.append(f"BF16 GSM8K accuracy {m['gsm8k_acc']:.3f} is below {GSM_MIN}")
    if not m["needle_acc"] >= NEEDLE_MIN:
        bad.append(f"BF16 needle accuracy {m['needle_acc']:.3f} is below {NEEDLE_MIN}")
    if not (math.isfinite(m["ref_nll"]) and m["ref_nll"] <= REF_NLL_MAX):
        bad.append(f"BF16 reference NLL {m['ref_nll']:.3f} nats per token is above {REF_NLL_MAX}")
    return bad, m


def preflight_problems(lps: dict, ev: dict, n_layers: int, length: int, arms=ARMS) -> list[str]:
    """The pre-flight on the probe (one prompt, one chunk): V5 and V7 for every arm, full token counts, finite values,
    DET equal to BF16 bit for bit or within DET_TOL, and every changed arm not bit-identical to BF16 when DET is."""
    bad = [] if n_layers else ["no hooked layer"]
    for arm in arms:
        v5, v7 = an55.engine_problems(arm, n_layers, ev)
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
    if det_same:
        bad += [f"{a} gives the BF16 log-probabilities bit for bit (the engine may ignore its hook)"
                for a in arms if a != "BF16" and np.array_equal(lps[a], lps["BF16"])]
    return bad


def expected_tokens(data: dict, outs: dict | None) -> dict:
    """Workload -> the tokens every hooked layer must see at least (README 4.5): each scored prompt once; each
    generated sequence its prompt and all but its last output token (preemption can add recomputed tokens)."""
    exp = {"PROBE": len(data["health"][0]["ids"]), "SCORE": sum(len(g["score_ids"]) for g in data["gsm8k"])}
    for h in data["health"][1:]:
        exp[h["id"]] = len(h["ids"])
    if outs:
        for task in ("gsm8k", "needle"):
            items = {x["id"]: x for x in data[task]}
            got = outs.get(task, {})
            exp[task.upper()] = sum(len(items[i]["ids"]) + max(len(o["ids"]) - 1, 0) for i, o in got.items())
    return exp


def det_check(outs: dict, arrays: dict) -> dict:
    """V3: DET (BF16 again with the hooks removed, the same workload in the same order) reproduces BF16: every answer
    has the same token ids, and the mean absolute log-probability difference is at most DET_TOL."""
    if "DET" not in outs or "BF16" not in outs:
        return {"ok": False, "why": "no DET or no BF16 outputs"}
    same = n = 0
    for task in ("gsm8k", "needle"):
        base, det = outs["BF16"].get(task, {}), outs["DET"].get(task, {})
        for i, o in base.items():
            n += 1
            same += int(det.get(i, {}).get("ids") == o["ids"])
    a, b = arrays.get("DET", {}), arrays.get("BF16", {})
    keys = [k for k in b if k.endswith("_lp")]
    missing = [k for k in keys if k not in a or len(a[k]) != len(b[k])]
    diff = [np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64)) for k in keys if k not in missing]
    mean_abs = float(np.concatenate(diff).mean()) if diff else float("inf")
    max_abs = float(max(float(x.max()) for x in diff)) if diff else float("inf")
    ok = n > 0 and same == n and not missing and mean_abs <= DET_TOL
    return {"ok": ok, "answers": n, "same_ids": same, "missing": missing[:5], "mean_abs_lp": mean_abs,
            "max_abs_lp": max_abs, "bit_identical": ok and max_abs == 0.0}


def validity(res: Path, data: dict, main: dict, outs: dict, graded: dict, ref: dict, rehearsal: bool,
             arrays: dict | None = None) -> tuple:
    v = {k: [] for k in V_KEYS}
    excluded: dict[str, list[str]] = {}
    stub = bool(main.get("stub"))
    gpu = str(main.get("gpu", ""))
    if not (stub or (("A100" in gpu or "H100" in gpu) and "80GB" in gpu)):
        v["V1"].append(f"the GPU is {gpu!r}, not an A100 or H100 80GB")
    if main.get("status") != "ok" and not main.get("decision_arms_done"):
        v["V4"].append(f"run_main status {main.get('status')} before the decision arms finished: {main.get('errors')}")
    if not rehearsal:
        if main.get("data_sha256") != DATA_SHA256:
            v["V2"].append("data_e56.json is not the frozen file")
        if not stub and main.get("snapshot_revision") != REVISION:
            v["V2"].append(f"the model snapshot {main.get('snapshot_revision')} is not the pinned revision")
    if not main.get("tok_check", {}).get("ok"):
        v["V2"].append("the engine tokenizer does not give the frozen ids")
    if not stub:
        e = main.get("engine", {})
        if str(e.get("kv_cache_dtype")) != "auto" or e.get("prefix_caching") not in (False, "False"):
            v["V2"].append("the cache is not BF16 auto, or prefix caching is on")
        mode = str(e.get("compilation_mode"))   # E-55 A2: this build records CompilationMode.NONE as "0"
        if not (mode == "0" or "NONE" in mode) or "NONE" not in str(e.get("cudagraph_mode")):
            v["V2"].append("the engine is not eager")
        if e.get("max_num_batched_tokens") != ENGINE_TOKENS:
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
    layers = [h["layer"] for h in main.get("hooks", {}).get("hooked", [])]
    if not rehearsal and layers != list(GLOBAL_LAYERS):
        v["V3"].append(f"hooked layers {layers}, not the 10 global layers")
    if not layers:
        v["V3"].append("no hooked layer")
    n_layers = len(layers)
    counts = main.get("counts", {})
    for arm in [a for a in ARMS if a in main.get("arms", {})]:
        exp = expected_tokens(data, outs.get(arm))
        bad = []
        for work, n in exp.items():
            seen = [counts.get(f"{arm}|{work}|{lay}", 0) for lay in layers]
            if any(s < n for s in seen):
                bad.append(f"{arm} {work}: the hooks saw {min(seen) if seen else 0} tokens, at least {n} expected")
        if arm in DECISION:
            v["V3"] += bad
        elif bad:
            excluded.setdefault(arm, []).extend(bad)
    det = det_check(outs, arrays or {})
    if "DET" in main.get("arms", {}) and not det["ok"]:
        v["V3"].append(f"DET does not reproduce BF16: {det}")
    for arm in [a for a in ARMS if a in main.get("arms", {})]:
        v5, v7 = an55.engine_problems(arm, n_layers, main)
        if not main["arms"][arm].get("finite", False):
            v5.append(f"{arm}: a value is not finite")
        if arm in DECISION:
            v["V5"] += v5
            v["V7"] += v7
        elif v5 or v7:
            excluded.setdefault(arm, []).extend(v5 + v7)
    for arm in DECISION:
        if arm not in main.get("arms", {}):
            v["V4"].append(f"the decision arm {arm} did not run")
    if "BF16" in graded:
        v["V8"], _ = health_problems(graded["BF16"], ref.get("BF16", {}), data)
    return v, excluded


def decide(st: dict) -> tuple[str, str, str]:
    lo, hi = st["acc_ci"]
    h1 = "KILL1" if hi < 0 else ("GO1" if lo >= -M_ACC and hi <= M_ACC else "INCONCLUSIVE1")
    if st.get("dis_ci") is None:
        h2 = "NA2"
    else:
        dlo, dhi = st["dis_ci"]
        h2 = "GO2" if dhi <= M_DIS else ("WORSE2" if dlo > M_DIS else "INCONCLUSIVE2")
    nlo, nhi = st["nll_ci"]
    h3 = "KILL3" if nlo > KILL_NLL else ("GO3" if nlo >= -M_NLL and nhi <= M_NLL else "INCONCLUSIVE3")
    return h1, h2, h3


def summarize(res: Path, data_path: Path) -> dict:
    data = json.loads(data_path.read_text())
    main = load(res / "run_main.json", {})
    rehearsal = bool(main.get("rehearsal"))
    outs = {a: load(res / f"outputs_{a}.json") for a in ARMS + ("DET",) if (res / f"outputs_{a}.json").exists()}
    arrays = {p.stem: dict(np.load(p)) for p in sorted((res / "arrays").glob("*.npz"))}
    graded = {a: grade(data, o) for a, o in outs.items()}
    ref = {a: ref_nll(arr, data) for a, arr in arrays.items()}
    v, excluded = validity(res, data, main, outs, graded, ref, rehearsal, arrays)
    out = {"validity": v, "excluded": excluded, "rehearsal": rehearsal, "status": main.get("status"),
           "cut": main.get("cut", []), "step_errors": main.get("step_errors", []),
           "preflight": (main.get("preflight") or {}).get("problems"), "health": main.get("health"),
           "det": det_check(outs, arrays)}
    if any(v[k] for k in V_KEYS) and not rehearsal:
        out["decision"] = "INVALID"
        out["failed"] = [k for k in V_KEYS if v[k]]
        return out
    items = [g["id"] for g in data["gsm8k"]] + [n["id"] for n in data["needle"]]
    base = graded.get("BF16", {})
    st: dict = {"items": len(items), "accuracy": {}, "by_task": {}, "differ": {}, "same_text": {}, "nll": {}}
    for arm, gr in graded.items():
        if arm in excluded or not all(i in gr for i in items):
            continue
        st["accuracy"][arm] = float(np.mean([gr[i]["correct"] for i in items]))
        st["by_task"][arm] = {t: float(np.mean([gr[x["id"]]["correct"] for x in data[t]])) for t in ("gsm8k", "needle")}
        if arm != "BF16":
            st["differ"][arm] = float(np.mean([gr[i]["answer"] != base[i]["answer"] for i in items]))
            st["same_text"][arm] = float(np.mean([gr[i]["ids"] == base[i]["ids"] for i in items]))
    x = [base[i]["correct"] for i in items]
    y = [graded["BF16C"][i]["correct"] for i in items]
    st["acc_diff"], st["acc_ci"] = paired_diff(x, y)
    if "BF16R" in st["accuracy"]:
        dc = [graded["BF16C"][i]["answer"] != base[i]["answer"] for i in items]
        dr = [graded["BF16R"][i]["answer"] != base[i]["answer"] for i in items]
        st["dis_diff"], st["dis_ci"] = paired_diff(dr, dc)
        st["acc_diff_bf16r"], st["acc_ci_bf16r"] = paired_diff(x, [graded["BF16R"][i]["correct"] for i in items])
    else:
        st["dis_diff"], st["dis_ci"] = None, None
    rb = ref.get("BF16", {})
    for arm, r in ref.items():
        if arm in ("BF16", "DET") or arm in excluded or set(r) != set(rb):
            continue
        ids = sorted(rb)
        s = np.array([rb[i][0] - r[i][0] for i in ids])     # the NLL change: positive = the arm is less likely
        n = np.array([rb[i][1] for i in ids], dtype=np.float64)
        d, ci = ratio_interval(s, n)
        st["nll"][arm] = {"delta": d, "ci": ci}
    st["nll_ci"] = st["nll"]["BF16C"]["ci"]
    st["base_ref_nll"] = health_problems(base, rb, data)[1]["ref_nll"]
    health = {}
    for h in data["health"]:
        lp = arrays.get("BF16", {}).get(f"{h['id']}_lp")
        if lp is not None:
            health[h["source"]] = float(-np.asarray(lp, np.float64).mean())
    st["health_raw_nll"] = health
    out["stats"] = st
    h1, h2, h3 = decide(st)
    out.update({"decision": f"{h1} {h2} {h3}", "H1": h1, "H2": h2, "H3": h3})
    return out


def line(s: dict) -> str:
    probs = "; ".join(f"{k}: {p[:2]}" for k, p in s["validity"].items() if p)
    if s["decision"] == "INVALID":
        return f"INVALID ({', '.join(s['failed'])}): {probs}"[:2500]
    st = s["stats"]
    acc = ", ".join(f"{a} {x:.4f}" for a, x in st["accuracy"].items())
    dif = ", ".join(f"{a} {x:.4f}" for a, x in st["differ"].items())
    nll = "; ".join(f"{a} {v['delta']:+.5f} [{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}]" for a, v in st["nll"].items())
    dis = (f"; differ BF16C - BF16R {st['dis_diff']:+.4f} [{st['dis_ci'][0]:+.4f}, {st['dis_ci'][1]:+.4f}]"
           if st.get("dis_ci") else "")
    pre = "REHEARSAL " if s["rehearsal"] else ""
    tail = f"; validity problems: {probs}" if probs else ""
    return (f"{pre}{s['decision']}: accuracy BF16C - BF16 {st['acc_diff']:+.4f} [{st['acc_ci'][0]:+.4f}, "
            f"{st['acc_ci'][1]:+.4f}] (margin {M_ACC}){dis}; accuracy {acc}; answer differs from BF16: {dif}; "
            f"reference NLL change {nll} (BF16 {st['base_ref_nll']:.4f}); raw-text NLL {st['health_raw_nll']}; "
            f"cut {[c['step'] for c in s['cut']]}{tail}")[:3000]


def manifest(s: dict, res: Path, design_commit: str, run_id: str) -> dict:
    """The ledger manifest of the run (MEASUREMENT.md Section 5): one value per metric."""
    main = load(res / "run_main.json", {})
    st = s.get("stats", {})
    metrics = {f"acc_{a}": {"values": [x], "unit": "fraction", "higher_is_better": True}
               for a, x in st.get("accuracy", {}).items()}
    metrics.update({f"differ_{a}": {"values": [x], "unit": "fraction", "higher_is_better": False}
                    for a, x in st.get("differ", {}).items()})
    metrics.update({f"dnll_ref_{a}": {"values": [x["delta"]], "unit": "nats/token", "higher_is_better": False}
                    for a, x in st.get("nll", {}).items()})
    return {"run_id": run_id, "experiment": "E-56", "idea": "I-182",
            "status": "ok" if s.get("decision") != "INVALID" else "failed", "design_commit": design_commit,
            "config": main.get("engine_kwargs", {}), "started_at": main.get("started_at"),
            "finished_at": main.get("finished_at"),
            "hardware": {"gpu_name": main.get("gpu"), "gpu_count": 1, "driver": None, "cuda": None},
            "software": {"vllm": main.get("engine", {}).get("vllm")}, "metrics": metrics,
            "results_path": str(res), "notes": f"decision={s['decision']}; {line(s)}"[:2000]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=HERE / "data_e56.json")
    ap.add_argument("--manifest-out", type=Path)
    ap.add_argument("--design-commit", default="")
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()
    s = summarize(args.res, args.data)
    (args.res / "summary.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    if args.manifest_out:
        m = manifest(s, args.res, args.design_commit, args.run_id or args.res.name)
        args.manifest_out.write_text(json.dumps(m, indent=1) + "\n")
    print(line(s))


if __name__ == "__main__":
    main()

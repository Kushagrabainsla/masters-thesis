"""E-27 decision script. Frozen before the run (amendment A1). Reads the result JSON files of
e27_run.py and applies the predeclared rules. Prints a report and writes e27_decision.json.

Rules (amendment A1, heuristics):
  (a) M1: A = sqrt(sum E(C3) / sum E(C1)) over evaluation windows (layer-local error energy).
      KILL if A > 1/1.5. The 95% CI comes from a bootstrap of windows stratified by corpus.
  Headroom: d0 = NLL(C1) - NLL(C0), paired per window, in nats. INCONCLUSIVE if mean d0 < 0.01
      or its 95% CI lower bound <= 0.
  (b) M2: d = NLL(C1) - NLL(C3). KILL if mean d < 0.01 or its 95% CI lower bound <= 0.
  GO: none of the above fires and mean NLL(C3) - NLL(C5) <= 0.01.
  Otherwise PIVOT.
  Attribution (report): the gain is a smoothness effect only if the dither control C1d and the
      random-order control C4r each capture less than half of the C3 energy gain.
  M3 passkey: report only, McNemar exact test of C1 vs C3.
"""

import argparse
import json
import math
import os
import random

B = 10000
SEED = 27
A_MAX = 1 / 1.5
NATS = 0.01


def stratified_boot(values_by_corpus, stat, b=B, seed=SEED):
    rng = random.Random(seed)
    out = []
    for _ in range(b):
        sample = {c: [rng.choice(v) for _ in v] for c, v in values_by_corpus.items()}
        out.append(stat(sample))
    out.sort()
    return out[int(0.025 * b)], out[int(0.975 * b) - 1]


def group(items, corpora):
    g = {}
    for x, c in zip(items, corpora):
        g.setdefault(c, []).append(x)
    return g


def mean(xs):
    return sum(xs) / len(xs)


def mcnemar(a_ok, b_ok):
    bb = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    cc = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return bb, cc, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory with e27_*.json")
    args = ap.parse_args()
    s1 = json.load(open(os.path.join(args.out, "e27_step1.json")))
    pp = json.load(open(os.path.join(args.out, "e27_ppl.json")))
    pk_path = os.path.join(args.out, "e27_passkey.json")
    pk = json.load(open(pk_path)) if os.path.exists(pk_path) else None
    assert pp["manifest"] == s1["manifest"] and pp["gates"] == s1["gates"]
    if pk:
        assert pk["manifest"] == s1["manifest"] and pk["gates"] == s1["gates"]

    corp = [w["corpus"] for w in s1["eval_windows"]]
    assert pp["corpus"] == corp
    assert len(corp) == 18 and all(corp.count(c) == 6 for c in ("prose", "code", "agent"))
    recs = s1["eval_records"]
    names = list(recs[0]["err_energy"].keys())
    E = {n: [sum(map(sum, r["err_energy"][n])) for r in recs] for n in names}
    rep = {"manifest": s1["manifest"], "gate_fraction": s1["gate_fraction"],
           "gate_split_half_agreement": s1["gate_split_half_agreement"],
           "smoothness_diagnostic": {k: {kk: vv for kk, vv in v.items() if kk != "G"}
                                     for k, v in s1["smoothness_diagnostic"].items()}}

    # M1, rule (a)
    pairs = group(list(zip(E["C3"], E["C1"])), corp)
    amp = lambda s: math.sqrt(sum(x for v in s.values() for x, _ in v) / sum(y for v in s.values() for _, y in v))
    A = amp(pairs)
    A_ci = stratified_boot(pairs, amp)
    gains = {n: 1 - sum(E[n]) / sum(E["C1"]) for n in names}
    per_corpus_A = {c: math.sqrt(sum(x for x, _ in v) / sum(y for _, y in v)) for c, v in pairs.items()}
    # secondary: per (layer, head) energy ratio C3/C1 summed over windows
    L, H = len(recs[0]["err_energy"]["C1"]), len(recs[0]["err_energy"]["C1"][0])
    cell = []
    for l in range(L):
        for h in range(H):
            num = sum(r["err_energy"]["C3"][l][h] for r in recs)
            den = sum(r["err_energy"]["C1"][l][h] for r in recs)
            cell.append(num / den if den > 0 else float("nan"))
    cell.sort()
    q = lambda f: cell[int(f * (len(cell) - 1))]
    rep["M1"] = {"A": A, "A_ci95": A_ci, "A_per_corpus": per_corpus_A, "energy_gain_vs_C1": gains,
                 "cell_ratio_quartiles": [q(0.25), q(0.5), q(0.75)]}
    rule_a = A > A_MAX

    # M2
    nll = pp["nll"]
    pc = pp["corpus"]
    def paired(a, b):
        d = [x - y for x, y in zip(nll[a], nll[b])]
        g = group(d, pc)
        return mean(d), stratified_boot(g, lambda s: mean([x for v in s.values() for x in v])), \
            {c: mean(v) for c, v in g.items()}
    d0 = paired("C1", "C0")
    d = paired("C1", "C3")
    g5 = paired("C3", "C5")
    extra = {}
    for a, b in (("C1", "C2"), ("C1", "C1d"), ("C1", "C4"), ("C1", "C4r"), ("C1u", "C1"),
                 ("C1z", "C3z"), ("C3z", "C5z")):
        if a in nll and b in nll:
            extra[f"{a}-{b}"] = paired(a, b)
    rep["M2"] = {"headroom_C1-C0": d0, "gain_C1-C3": d, "C3-C5": g5, "other_pairs": extra,
                 "mean_nll": {n: mean(v) for n, v in nll.items()}}
    inconclusive = d0[0] < NATS or d0[1][0] <= 0
    rule_b = d[0] < NATS or d[1][0] <= 0
    go_bit = g5[0] <= NATS

    # M3 (report only)
    if pk:
        ok = pk["correct"]
        rep["M3"] = {"accuracy": {n: mean([1.0 if x else 0.0 for x in v]) for n, v in ok.items()}}
        if "C1" in ok and "C3" in ok:
            rep["M3"]["mcnemar_C1_vs_C3"] = dict(zip(("C1_only", "C3_only", "p"), mcnemar(ok["C1"], ok["C3"])))

    # Attribution (report)
    g3 = gains.get("C3", 0.0)
    rep["attribution"] = {
        "gain_C3": g3, "gain_C1d": gains.get("C1d"), "gain_C4r": gains.get("C4r"),
        "smoothness_effect": g3 > 0 and gains.get("C1d", 0) < 0.5 * g3 and gains.get("C4r", 0) < 0.5 * g3,
    }

    if rule_a:
        verdict = "KILL (rule a: A > 1/1.5)"
    elif inconclusive:
        verdict = "INCONCLUSIVE (no perplexity headroom between C1 and C0)"
    elif rule_b:
        verdict = "KILL (rule b: perplexity gain of C3 over C1 under 0.01 nats or not significant)"
    elif go_bit:
        verdict = "GO (C3 within 0.01 nats of C5: one bit saved on V)"
    else:
        verdict = "PIVOT (C3 beats C1 but does not reach C5)"
    rep["rules"] = {"a": rule_a, "inconclusive": inconclusive, "b": rule_b, "go_bit": go_bit}
    rep["verdict"] = verdict
    json.dump(rep, open(os.path.join(args.out, "e27_decision.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

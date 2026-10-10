"""E-28 frozen decision script, version 2 (amendment A1). Frozen before stage 1 runs.

All thresholds are project heuristics (adapted from R-14 and R-15), not literature standards.
A point is (mix, cache, load, spike_frac). Controls do not depend on the spike size.
Usage: python3 -I e28_decide.py   (reads results/e28_fcfs.csv and results/e28_cap.csv)
"""
import csv
import os
import random
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COLLAPSE_RATIO = 0.7      # spike goodput below 70% of control goodput
STABLE_RATIO = 0.9        # control late goodput at least 90% of control pre goodput
SEED_MAJORITY = 7         # of 10 paired seeds, in both the post and the late window
RETRY_SHARE = 0.8         # K2
FIXED_LOSS_KILL = 0.10    # K3 kill
FIXED_LOSS_GO = 0.20      # K3 go
GO_MIXES = 3
RESTORE_GOODPUT = 0.8     # GO-B: the oracle cap must reach 0.8 x control late goodput ...
RESTORE_SESSIONS = 0.9    # ... and 0.9 x control session success
SEL_SEEDS = set(range(1, 6))     # cap selection
EVAL_SEEDS = set(range(6, 11))   # cap evaluation


def load(name):
    return list(csv.DictReader(open(os.path.join(RES, f"e28_{name}.csv"))))


def g(r, w):
    return float(r[f"{w}_turn_ok"])


def succ(r):
    """Session success share over the post and late windows (ended sessions only)."""
    ok = float(r["post_sess_ok"]) + float(r["late_sess_ok"])
    bad = float(r["post_sess_fail"]) + float(r["late_sess_fail"])
    return ok / (ok + bad) if ok + bad > 0 else 1.0


def main():
    fc = load("fcfs")
    spike, ctrl = defaultdict(dict), defaultdict(dict)
    for r in fc:
        base = (r["mix"], int(r["kv_capacity_tok"]), float(r["load"]))
        if r["kind"] == "ctrl":
            ctrl[base + (r["client"],)][int(r["seed"])] = r
        else:
            spike[base + (float(r["spike_frac"]), r["client"])][int(r["seed"])] = r
    m = lambda rows, w: st.mean(g(r, w) for r in rows)
    res = {}
    for k, d in sorted(spike.items()):
        mix, cache, L, sf, cl = k
        c = ctrl[(mix, cache, L, cl)]
        seeds = sorted(set(d) & set(c))
        sp, ct = [d[s] for s in seeds], [c[s] for s in seeds]
        stable = m(ct, "late") >= STABLE_RATIO * m(ct, "pre")
        paired = sum(g(a, "post") < COLLAPSE_RATIO * g(b, "post") and
                     g(a, "late") < COLLAPSE_RATIO * g(b, "late") for a, b in zip(sp, ct))
        col = (stable and m(sp, "post") < COLLAPSE_RATIO * m(ct, "post")
               and m(sp, "late") < COLLAPSE_RATIO * m(ct, "late") and paired >= SEED_MAJORITY)
        res[k] = dict(col=col, stable=stable, ctrl=c)
        print(f"{k}: stable={stable} collapse={col} paired={paired}/{len(seeds)} "
              f"ctrl pre/post/late={m(ct,'pre'):.3f}/{m(ct,'post'):.3f}/{m(ct,'late'):.3f} "
              f"spike post/late={m(sp,'post'):.3f}/{m(sp,'late'):.3f}")
    pts = lambda cl: [k[:4] for k, v in res.items() if k[4] == cl and v["col"]]
    pa, pb, pc = pts("A"), pts("B"), pts("C")
    n_stable = {cl: sum(v["stable"] for k, v in res.items() if k[4] == cl) for cl in "ABC"}
    print(f"\ncollapse points: A={len(pa)} B={len(pb)} C={len(pc)}; stable controls: {n_stable}")

    verdict = {}
    verdict["K1"] = "KILL (no collapse)" if not pa and not pc else "PASS"
    if pa:
        share_c = sum(p in pc for p in pa) / len(pa)
        share_b = sum(p in pb for p in pa) / len(pa)
        print(f"K2: share of A points that also collapse under C: {share_c:.2f} (B: {share_b:.2f})")
        verdict["K2"] = "KILL (retries not needed)" if share_c >= RETRY_SHARE else "PASS"
    else:
        verdict["K2"] = "KILL (retries not needed: collapse only without retries)" if pc else "n/a"

    if pa:
        cp = load("cap")
        sel, ev = defaultdict(list), defaultdict(list)
        evrows = defaultdict(list)
        for r in cp:
            k = (r["mix"], int(r["kv_capacity_tok"]), float(r["load"]), float(r["spike_frac"]),
                 int(r["session_cap"]))
            (sel if int(r["seed"]) in SEL_SEEDS else ev)[k].append(g(r, "late"))
            if int(r["seed"]) in EVAL_SEEDS:
                evrows[k].append(r)
        caps = sorted({int(r["session_cap"]) for r in cp})
        loads = sorted({float(r["load"]) for r in cp})
        sfs = sorted({float(r["spike_frac"]) for r in cp})
        caches = sorted({int(r["kv_capacity_tok"]) for r in cp})
        fixed = {c: max(caps, key=lambda cap: st.mean(
            st.mean(sel[("M1_all", c, L, sf, cap)]) for L in loads for sf in sfs)) for c in caches}
        print("fixed cap per cache (tuned on M1_all, selection seeds):", fixed)
        loss, not_restored, per_mix = [], 0, defaultdict(list)
        rng = random.Random(0)
        for p in pa:
            mix, c, L, sf = p
            best = max(caps, key=lambda cap: st.mean(sel[p + (cap,)]))
            gb, gf = st.mean(ev[p + (best,)]), st.mean(ev[p + (fixed[c],)])
            lo = 1 - gf / gb if gb > 0 else 0.0
            ctrl_a = [r for s, r in res[p + ("A",)]["ctrl"].items() if s in EVAL_SEEDS]
            c_late = st.mean(g(r, "late") for r in ctrl_a)
            c_succ = st.mean(succ(r) for r in ctrl_a)
            o_succ = st.mean(succ(r) for r in evrows[p + (best,)])
            restored = gb >= RESTORE_GOODPUT * c_late and o_succ >= RESTORE_SESSIONS * c_succ
            er = evrows[p + (best,)]
            adm = sum(float(r["late_admitted"]) for r in er)
            gw = sum(float(r["late_gw_wait"]) for r in er) / adm if adm else 0.0
            not_restored += not restored
            loss.append(lo)
            per_mix[mix].append(lo)
            # Bootstrap over evaluation seeds for the loss (reported only).
            xb, xf = ev[p + (best,)], ev[p + (fixed[c],)]
            bs = sorted(1 - st.mean(rng.choices(xf, k=len(xf))) /
                        max(st.mean(rng.choices(xb, k=len(xb))), 1e-9) for _ in range(500))
            print(f"  {p}: oracle cap={best} G={gb:.3f} fixed cap={fixed[c]} G={gf:.3f} "
                  f"loss={lo:.2f} [{bs[12]:.2f}, {bs[487]:.2f}] ctrl late={c_late:.3f} "
                  f"succ oracle/ctrl={o_succ:.2f}/{c_succ:.2f} gw_wait={gw:.1f}s restored={restored}")
        small = sum(x <= FIXED_LOSS_KILL for x in loss) / len(loss)
        go_mixes = [mx for mx, v in per_mix.items() if st.mean(v) > FIXED_LOSS_GO]
        nr = not_restored / len(pa)
        print(f"K3: share not restored by the oracle cap: {nr:.2f}; share with fixed-cap loss <= "
              f"{FIXED_LOSS_KILL}: {small:.2f}; mixes with mean loss > {FIXED_LOSS_GO}: {go_mixes}")
        if nr >= 0.5:
            verdict["K3"] = "GO-B (a static cap does not fix the collapse)"
        elif small >= 0.8:
            verdict["K3"] = "KILL (a fixed cap is enough)"
        elif len(go_mixes) >= GO_MIXES:
            verdict["K3"] = "GO (a fixed cap loses > 20% in >= 3 mixes)"
        else:
            verdict["K3"] = "PIVOT"
    else:
        verdict["K3"] = "n/a"
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()

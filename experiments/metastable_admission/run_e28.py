"""E-28 runner (version 2, amendment A1). Stages run in order; each writes results/e28_<stage>.csv.

  python3 -I run_e28.py calibrate  # stage 0a: G* and N* (best cap) per variant, mix, cache
  python3 -I run_e28.py knee       # stage 0b: largest stable load of the uncapped control
  python3 -I run_e28.py fcfs       # stage 1: default engine, spike and control, clients A, B, C
  python3 -I run_e28.py cap        # stage 2: static session caps on the stage 1 spike runs
  python3 -I run_e28.py sens       # stage 3: constant sensitivity on M1 (reported, not decided)

e28_decide.py reads the stage 1 and stage 2 files. It is frozen before stage 1 runs.
"""
from __future__ import annotations

import csv
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import Params, Sim, clean_calls, load_sessions  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "session_calls.csv")
OUT = os.path.join(HERE, "results")

# ----------------------------------------------------------------------------- fixed design
SEEDS = list(range(1, 11))
CAL_SEEDS = list(range(1, 6))
CACHES = [190_000, 380_000]
CAPS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
KNEE_GRID = [round(0.30 + 0.05 * i, 2) for i in range(15)]      # 0.30 .. 1.00 x G*
LOAD_FRACS = [0.6, 0.7, 0.8, 0.9, 1.0]                          # x knee
SPIKE_FRACS = [0.5, 1.0]        # extra sessions in the spike, as a fraction of N*
CLIENTS = {"A": (60.0, 2), "B": (60.0, 0), "C": (600.0, 2)}      # (timeout_s, max_retries)
SPIKE_START, SPIKE_LEN, HORIZON = 2400.0, 60.0, 6000.0
WIN = {"pre": (1800, 2400), "post": (3060, 3660), "late": (5400, 6000)}
CAL_WIN = (3600, 6000)


def _median_prompt(cc):
    v = sorted(int(c["in_tok"]) for c in cc)
    return v[len(v) // 2]


MIXES = {   # keep() receives the cleaned calls of one session (sim.clean_calls)
    "M1_all": dict(keep=None, gap_scale=1.0),
    "M2_claude_code": dict(keep=lambda c: c[0]["harness"] == "claude_code", gap_scale=1.0),
    "M3_other_harness": dict(keep=lambda c: c[0]["harness"] != "claude_code", gap_scale=1.0),
    "M4_short_ctx": dict(keep=lambda c: _median_prompt(c) <= 20000, gap_scale=1.0),
    "M5_slow_tools": dict(keep=None, gap_scale=4.0),
}
BASE = Params()
VARIANTS = {   # stage 3 only, on M1_all and the 380k cache
    "base": {},
    "out_clip_1024": dict(out_clip=1024),
    "prefill_x0.5": dict(prefill_s_per_tok=BASE.prefill_s_per_tok * 2,
                         attn_s_per_tok_ctx=BASE.attn_s_per_tok_ctx * 2),   # half as fast
    "prefill_x2": dict(prefill_s_per_tok=BASE.prefill_s_per_tok / 2,
                       attn_s_per_tok_ctx=BASE.attn_s_per_tok_ctx / 2),     # twice as fast
    "decode_x0.5": dict(decode_s_per_ctx_tok=BASE.decode_s_per_ctx_tok * 2),
    "decode_x2": dict(decode_s_per_ctx_tok=BASE.decode_s_per_ctx_tok / 2),
    "gap_x0.25": dict(gap_scale=0.25),
}

_POOLS: dict = {}


def pool(mix):
    if mix not in _POOLS:
        _POOLS[mix] = load_sessions(DATA, BASE.max_context_tok, MIXES[mix]["keep"])
    return _POOLS[mix]


def mean_turns(mix):
    return st.mean(len(s.turns) for s in pool(mix)[0])


def params(mix, cache, variant="base", **kw):
    v = dict(gap_scale=MIXES[mix]["gap_scale"])
    v.update(VARIANTS[variant])
    if "gap_scale" in VARIANTS[variant]:
        v["gap_scale"] = MIXES[mix]["gap_scale"] * VARIANTS[variant]["gap_scale"]
    base = dict(kv_capacity_tok=cache, spike_start_s=SPIKE_START, spike_len_s=SPIKE_LEN,
                horizon_s=HORIZON, **v)
    base.update(kw)
    return replace(Params(), **base)


def win(bins, key, t0, t1):
    return sum(v[key] for k, v in bins.items() if t0 <= k * 10 < t1) / (t1 - t0)


KEYS = ("turn_ok", "timeouts", "sess_ok", "sess_fail", "shed", "admitted", "gw_wait",
        "arrivals", "preempt", "prefill_tok", "hit_tok", "busy")


def job(args):
    mix, p = args
    specs, gaps = pool(mix)
    bins = Sim(p, specs, gaps).run()
    row = {}
    wins = dict(WIN, cal=CAL_WIN)
    for name, (t0, t1) in wins.items():
        for key in KEYS:
            row[f"{name}_{key}"] = win(bins, key, t0, t1)
        row[f"{name}_active"] = bins.get(t1 // 10 - 1, {}).get("active", 0)
    series = [(k * 10, v["turn_ok"], v["active"], v["preempt"], v["timeouts"])
              for k, v in sorted(bins.items())]
    return row, series


def run_jobs(name, specs):
    """specs: list of (meta dict, mix, Params). Writes results/e28_<name>.csv and a series file."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"e28_{name}.csv")
    ser_path = os.path.join(OUT, f"e28_{name}_series.csv")
    with ProcessPoolExecutor(max(1, os.cpu_count() - 1)) as ex, \
            open(path, "w", newline="") as fh, open(ser_path, "w", newline="") as fs:
        w = None
        ws = csv.writer(fs)
        ws.writerow(["run", "t", "turn_ok", "active", "preempt", "timeouts"])
        res = ex.map(job, [(m, p) for _, m, p in specs], chunksize=2)
        for i, ((meta, mix, p), (row, series)) in enumerate(zip(specs, res)):
            rec = {"run": i, **meta, "mix": mix, **asdict(p), **row}
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(rec))
                w.writeheader()
            w.writerow(rec)
            for t in series:
                ws.writerow([i, *t])
            if i % 200 == 0:
                print(f"{name}: {i}/{len(specs)}", flush=True)
    print("wrote", path, len(specs), "runs")


def read(name):
    return list(csv.DictReader(open(os.path.join(OUT, f"e28_{name}.csv"))))


def calibration():
    """(variant, mix, cache) -> (G*, N*): the best mean cal-window goodput over caps, and its cap."""
    acc = {}
    for r in read("calibrate"):
        k = (r["variant"], r["mix"], int(r["kv_capacity_tok"]))
        acc.setdefault(k, {}).setdefault(int(r["session_cap"]), []).append(float(r["cal_turn_ok"]))
    out = {}
    for k, caps in acc.items():
        best = max(caps, key=lambda c: st.mean(caps[c]))
        out[k] = (st.mean(caps[best]), best)
    return out


KNEE_MEAN = 0.85            # mean-based stability threshold (unit test: >= 90% pass rate)
KNEE_HEALTH_TIMEOUT = 0.02   # late timeouts per completed turn
KNEE_HEALTH_PREEMPT = 0.05   # late preemptions per completed turn


def knee_ok(rows, offered):
    """One knee grid load is stable if (R-15 N1: mean-based tests only, plus health checks):
    mean late goodput >= 0.85 x offered, mean late >= 0.85 x mean pre, and the late timeout and
    preemption rates per completed turn are small."""
    late = st.mean(float(r["late_turn_ok"]) for r in rows)
    pre = st.mean(float(r["pre_turn_ok"]) for r in rows)
    to = st.mean(float(r["late_timeouts"]) for r in rows)
    pr = st.mean(float(r["late_preempt"]) for r in rows)
    return (late >= KNEE_MEAN * offered and late >= KNEE_MEAN * pre
            and to <= KNEE_HEALTH_TIMEOUT * late and pr <= KNEE_HEALTH_PREEMPT * late)


def knee_from(by, g):
    """by: frac -> rows. The knee is the last passing load before the first two consecutive
    failures (KNEE_GRID[0] if nothing passes)."""
    knee, fails = None, 0
    for f in sorted(by):
        if knee_ok(by[f], f * g):
            knee, fails = f, 0
        else:
            fails += 1
            if fails == 2:
                break
    return knee if knee is not None else KNEE_GRID[0]


def knees():
    """(variant, mix, cache) -> knee fraction of G*, from the uncapped control (client C)."""
    cal = calibration()
    acc = {}
    for r in read("knee"):
        k = (r["variant"], r["mix"], int(r["kv_capacity_tok"]))
        acc.setdefault(k, {}).setdefault(float(r["frac"]), []).append(r)
    return {k: knee_from(by, cal[k][0]) for k, by in acc.items()}


def grid_points(variants, mixes, caches):
    """Yield (variant, mix, cache, load_frac, rate, spike_frac, spike_mult) for stages 1 to 3."""
    cal, kn = calibration(), knees()
    for v in variants:
        for mix in mixes:
            mt = mean_turns(mix)
            for cache in caches:
                g, nstar = cal[(v, mix, cache)]
                for lf in LOAD_FRACS:
                    rate = lf * kn[(v, mix, cache)] * g / mt
                    for sf in SPIKE_FRACS:
                        mult = 1 + sf * nstar / (rate * SPIKE_LEN)
                        thin = 1 + max(SPIKE_FRACS) * nstar / (rate * SPIKE_LEN)
                        yield v, mix, cache, lf, rate, sf, mult, thin


def main():
    stage = sys.argv[1]
    specs = []
    if stage == "calibrate":
        combos = [("base", m, c) for m in MIXES for c in CACHES]
        combos += [(v, "M1_all", 380_000) for v in VARIANTS if v != "base"]
        for v, mix, cache in combos:
            for cap in CAPS:
                for seed in CAL_SEEDS:
                    p = params(mix, cache, v, base_rate=1.0, spike_len_s=0.0, timeout_s=600.0,
                               max_retries=0, policy="cap", session_cap=cap, seed=seed)
                    specs.append(({"stage": stage, "variant": v}, mix, p))
    elif stage == "knee":
        cal = calibration()
        for (v, mix, cache), (g, _) in cal.items():
            for f in KNEE_GRID:
                for seed in SEEDS:
                    p = params(mix, cache, v, base_rate=f * g / mean_turns(mix), spike_len_s=0.0,
                               timeout_s=600.0, max_retries=2, seed=seed)
                    specs.append(({"stage": stage, "variant": v, "frac": f}, mix, p))
    elif stage in ("fcfs", "cap", "sens"):
        if stage == "sens":
            pts = grid_points([v for v in VARIANTS if v != "base"], ["M1_all"], [380_000])
        else:
            pts = grid_points(["base"], list(MIXES), CACHES)
        seen_ctrl = set()
        for v, mix, cache, lf, rate, sf, mult, thin in pts:
            meta = {"stage": stage, "variant": v, "load": lf, "spike_frac": sf}
            for seed in SEEDS:
                if stage == "cap":
                    to, rt = CLIENTS["A"]
                    for cap in CAPS:
                        p = params(mix, cache, v, base_rate=rate, spike_mult=mult, thin_mult=thin, timeout_s=to,
                                   max_retries=rt, seed=seed, policy="cap", session_cap=cap)
                        specs.append(({**meta, "kind": "spike", "client": "A"}, mix, p))
                    continue
                clients = CLIENTS if stage == "fcfs" else {"A": CLIENTS["A"], "C": CLIENTS["C"]}
                for cl, (to, rt) in clients.items():
                    p = params(mix, cache, v, base_rate=rate, spike_mult=mult, thin_mult=thin, timeout_s=to,
                               max_retries=rt, seed=seed)
                    specs.append(({**meta, "kind": "spike", "client": cl}, mix, p))
                    ck = (v, mix, cache, lf, cl, seed)
                    if ck not in seen_ctrl:   # the control does not depend on the spike size
                        seen_ctrl.add(ck)
                        c = replace(p, spike_len_s=0.0)
                        specs.append(({**meta, "spike_frac": "ctrl", "kind": "ctrl",
                                       "client": cl}, mix, c))
    else:
        raise SystemExit(f"unknown stage {stage}")
    run_jobs(stage, specs)


if __name__ == "__main__":
    main()

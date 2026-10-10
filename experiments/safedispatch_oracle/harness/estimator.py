"""Margin-stratified estimator of the greedy flip rate, and its coverage test (C4, E-23).

Idea: flips between a reference and a candidate backend concentrate at positions where the
reference top1 minus top2 logit margin is small. Stratify positions by that margin, evaluate
the candidate on a sample that over-weights small margins, and estimate the overall flip
rate with an interval. The pilot checks the interval against exhaustive evaluation.

All parameters are the ones predeclared in RESEARCH_JOURNAL.md, E-23.
"""
import numpy as np

EDGES = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf])
MIDS = np.array([0.025, 0.075, 0.175, 0.375, 0.75, 1.5, 3.0])
WEIGHTS = 1.0 / (MIDS + 0.1)
Z95 = 1.959963984540054


def stratify(margins: np.ndarray, edges: np.ndarray = EDGES) -> np.ndarray:
    """Bin index per margin. Bin b covers [edges[b], edges[b+1])."""
    idx = np.searchsorted(edges, margins, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def allocate(sizes: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    """Integer allocation proportional to sizes * weights, capped at sizes, summing to n."""
    sizes = np.asarray(sizes, dtype=float)
    n = int(min(n, sizes.sum()))
    target = sizes * weights
    alloc = np.zeros_like(sizes)
    active = sizes > 0
    remaining = float(n)
    while remaining > 1e-9 and active.any():
        share = np.where(active, target, 0.0)
        share = share / share.sum() * remaining
        cap = sizes - alloc
        over = active & (share >= cap - 1e-12)
        if over.any():
            alloc[over] = sizes[over]
            remaining -= cap[over].sum()
            active[over] = False
        else:
            alloc += share
            remaining = 0.0
    base = np.floor(alloc + 1e-9)
    left = int(n - base.sum())
    if left > 0:
        frac = alloc - base
        room = sizes - base
        order = np.argsort(-frac)
        for i in order:
            if left == 0:
                break
            if room[i] >= 1:
                base[i] += 1
                left -= 1
    return base.astype(int)


def sample_stratified(strata: np.ndarray, alloc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    picks = []
    for b, nb in enumerate(alloc):
        if nb <= 0:
            continue
        members = np.nonzero(strata == b)[0]
        picks.append(rng.choice(members, size=int(nb), replace=False))
    return np.concatenate(picks) if picks else np.array([], dtype=int)


def jeffreys_interval(k: np.ndarray, n: np.ndarray, N: np.ndarray, rng: np.random.Generator,
                      draws: int = 2000, return_ucb: bool = False):
    """95% interval for sum_b (N_b / N) p_b. Bins fully observed contribute no uncertainty."""
    total = N.sum()
    est = np.zeros(draws)
    point = 0.0
    for kb, nb, Nb in zip(k, n, N):
        if Nb == 0:
            continue
        w = Nb / total
        if nb >= Nb:
            est += w * (kb / Nb)
            point += w * (kb / Nb)
        elif nb == 0:
            est += w * rng.beta(0.5, 0.5, size=draws)
            point += w * 0.5
        else:
            est += w * rng.beta(kb + 0.5, nb - kb + 0.5, size=draws)
            point += w * (kb / nb)
    lo, hi = np.percentile(est, [2.5, 97.5])
    if return_ucb:
        return float(point), float(lo), float(hi), float(np.percentile(est, 95))
    return float(point), float(lo), float(hi)


def betabinom_interval(k: np.ndarray, n: np.ndarray, N: np.ndarray, rng: np.random.Generator,
                       draws: int = 4000, return_ucb: bool = False):
    """Amendment A1 (RESEARCH_JOURNAL.md, E-23). Finite-population interval for the flip rate.

    Sampled positions are counted exactly. For each stratum the flips among the m = N_b - n_b
    unsampled positions are drawn from a Beta-Binomial predictive with Jeffreys parameters
    (k_b + 0.5, n_b - k_b + 0.5). Zero stays a possible total for unsampled strata, which the
    rate-based Jeffreys interval could not represent.
    """
    k = np.asarray(k); n = np.asarray(n); N = np.asarray(N)
    total = np.full(draws, float(k.sum()))
    point = float(k.sum())
    for kb, nb, Nb in zip(k, n, N):
        m = int(Nb - nb)
        if m <= 0:
            continue
        if nb == 0:
            prob = rng.beta(0.5, 0.5, size=draws)
            point += m * 0.5
        else:
            prob = rng.beta(kb + 0.5, nb - kb + 0.5, size=draws)
            point += m * kb / nb
        total += rng.binomial(m, prob)
    rate = total / N.sum()
    lo, hi = np.percentile(rate, [2.5, 97.5])
    if return_ucb:
        return float(point / N.sum()), float(lo), float(hi), float(np.percentile(rate, 95))
    return float(point / N.sum()), float(lo), float(hi)


def wilson_interval(k: int, n: int, z: float = Z95):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def coverage_experiment(margins: np.ndarray, flips: np.ndarray, budget_frac: float = 0.25,
                        n_rep: int = 500, seed: int = 0, draws: int = 2000,
                        interval: str = "jeffreys_rate") -> dict:
    """Simulate the estimator against exhaustive flip data.

    margins, flips: one entry per (prompt, position). flips is boolean.
    """
    margins = np.asarray(margins, dtype=float)
    flips = np.asarray(flips, dtype=bool)
    N_total = len(margins)
    truth = float(flips.mean())
    strata = stratify(margins)
    sizes = np.array([(strata == b).sum() for b in range(len(EDGES) - 1)])
    flips_per_bin = np.array([flips[strata == b].sum() for b in range(len(EDGES) - 1)])
    n_budget = int(round(budget_frac * N_total))
    alloc = allocate(sizes, WEIGHTS, n_budget)

    covered = 0
    ucb_miss, ucb_miss_srs, ucb_ratio, ucb_ratio_srs = 0, 0, [], []
    widths_strat, widths_srs, covered_srs = [], [], 0
    for r in range(n_rep):
        rng = np.random.default_rng(seed + r)
        pick = sample_stratified(strata, alloc, rng)
        k = np.array([flips[pick][strata[pick] == b].sum() for b in range(len(EDGES) - 1)])
        if interval == "beta_binomial":
            _, lo, hi, ucb = betabinom_interval(k, alloc, sizes, rng, 4000, return_ucb=True)
        elif interval == "jeffreys_rate":
            _, lo, hi, ucb = jeffreys_interval(k, alloc, sizes, rng, draws, return_ucb=True)
        else:
            raise ValueError(interval)
        ucb_miss += int(truth > ucb)
        ucb_ratio.append(ucb / truth if truth > 0 else np.nan)
        covered += int(lo <= truth <= hi)
        widths_strat.append(hi - lo)
        srs = rng.choice(N_total, size=n_budget, replace=False)
        _, wlo, whi = wilson_interval(int(flips[srs].sum()), n_budget)
        _, _, wucb = wilson_interval(int(flips[srs].sum()), n_budget, z=1.6448536485112363)
        ucb_miss_srs += int(truth > wucb)
        ucb_ratio_srs.append(wucb / truth if truth > 0 else np.nan)
        covered_srs += int(wlo <= truth <= whi)
        widths_srs.append(whi - wlo)

    return {
        "n_positions": N_total,
        "n_flips": int(flips.sum()),
        "truth_flip_rate": truth,
        "budget_frac": budget_frac,
        "n_budget": n_budget,
        "bin_edges": [float(x) if np.isfinite(x) else "inf" for x in EDGES],
        "bin_sizes": sizes.tolist(),
        "bin_flips": flips_per_bin.tolist(),
        "bin_alloc": alloc.tolist(),
        "n_rep": n_rep,
        "interval": interval,
        "coverage_stratified": covered / n_rep,
        "coverage_srs_wilson": covered_srs / n_rep,
        "mean_width_stratified": float(np.mean(widths_strat)),
        "mean_width_srs_wilson": float(np.mean(widths_srs)),
        "one_sided_95_ucb_miss_stratified": ucb_miss / n_rep,
        "one_sided_95_ucb_miss_srs_wilson": ucb_miss_srs / n_rep,
        "mean_ucb_over_truth_stratified": float(np.nanmean(ucb_ratio)),
        "mean_ucb_over_truth_srs_wilson": float(np.nanmean(ucb_ratio_srs)),
        "width_ratio_srs_over_stratified": float(np.mean(widths_srs) / max(np.mean(widths_strat), 1e-12)),
    }

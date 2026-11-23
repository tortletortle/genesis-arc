"""Seeded active-query and open-set benchmark."""
import numpy as np
from open_set import identify_open_set

def random_permutation(rng, n=8, fix_zero=False):
    if fix_zero:
        return (0, *tuple(map(int, rng.permutation(np.arange(1, n)))))
    return tuple(map(int, rng.permutation(n)))

def candidate_family(seed, n=8, count=4):
    rng = np.random.default_rng(seed)
    base = list(random_permutation(rng, n, fix_zero=True))
    candidates = [tuple(base)]
    for a, b in [(1, 2), (3, 4), (5, 6), (1, 3), (2, 5)]:
        c = base.copy()
        c[a], c[b] = c[b], c[a]
        if tuple(c) not in candidates:
            candidates.append(tuple(c))
        if len(candidates) == count:
            break
    while len(candidates) < count:
        c = random_permutation(rng, n, fix_zero=True)
        if c not in candidates:
            candidates.append(c)
    unknown = random_permutation(rng, n, fix_zero=True)
    while unknown in candidates:
        unknown = random_permutation(rng, n, fix_zero=True)
    return candidates, unknown, rng

def run_active(seeds: int, noises):
    rows = []
    for noise in noises:
        for seed in range(seeds):
            candidates, unknown, rng = candidate_family(231000 + seed)
            true_candidate = candidates[int(rng.integers(0, len(candidates)))]
            for truth_type, truth in [("in_set", true_candidate), ("out_of_set", unknown)]:
                for strategy in ("active", "fixed", "random"):
                    result = identify_open_set(
                        candidates, truth, noise=float(noise), confidence=0.995,
                        unknown_prior=0.1, unknown_threshold=0.95,
                        min_queries=6, max_queries=80, strategy=strategy,
                        seed=232000 + seed, validation_inputs=4,
                        verification_repeats=3)
                    rows.append({
                        "noise": float(noise), "seed": seed,
                        "truth_type": truth_type, "strategy": strategy,
                        "decision": result.decision, "identified": result.identified,
                        "correct": result.correct, "queries": result.queries,
                        "candidate_confidence": result.posterior_confidence,
                        "unknown_probability": result.unknown_probability})
    return rows

def summarize(active_rows):
    summary = []
    for noise in sorted({r["noise"] for r in active_rows}):
        for truth_type in ("in_set", "out_of_set"):
            for strategy in ("active", "fixed", "random"):
                subset = [r for r in active_rows if r["noise"] == noise and
                          r["truth_type"] == truth_type and r["strategy"] == strategy]
                n = len(subset)
                if not n:
                    continue
                if truth_type == "in_set":
                    success = sum(r["decision"] == "identified" and r["correct"] for r in subset) / n
                    false_reject = sum(r["decision"] == "rejected_unknown" for r in subset) / n
                    false_accept = 0.0
                else:
                    success = sum(r["decision"] == "rejected_unknown" for r in subset) / n
                    false_reject = 0.0
                    false_accept = sum(r["decision"] == "identified" for r in subset) / n
                summary.append({
                    "noise": noise, "truth_type": truth_type, "strategy": strategy,
                    "n": n, "correct_decision_rate": success,
                    "false_reject_rate": false_reject, "false_accept_rate": false_accept,
                    "mean_queries": float(np.mean([r["queries"] for r in subset])),
                    "median_queries": float(np.median([r["queries"] for r in subset]))})
    return summary

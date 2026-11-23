"""Open-set Bayesian rejection with held-out active-query validation."""
from __future__ import annotations
import math
import numpy as np
from typing import Sequence
from active import IdentificationResult, _channel_entropy, _entropy

Map = tuple[int, ...]

def _open_set_information(candidates: Sequence[Map], posterior: Sequence[float],
                          unknown_probability: float, x: int, noise: float) -> float:
    n = len(candidates[0])
    output = np.full(n, unknown_probability / n, dtype=float)
    for weight, candidate in zip(posterior, candidates):
        output += weight * noise / (n - 1)
        output[candidate[x]] += weight * (1.0 - noise - noise / (n - 1))
    expected = (1.0 - unknown_probability) * _channel_entropy(n, noise)
    expected += unknown_probability * math.log2(n)
    return max(0.0, _entropy(output) - expected)

def identify_open_set(candidates: Sequence[Map], true_map: Map, noise: float = 0.0,
                      confidence: float = 0.995, unknown_prior: float = 0.1,
                      unknown_threshold: float = 0.95, min_queries: int = 6,
                      max_queries: int = 100, strategy: str = "active",
                      seed: int = 0, validation_inputs: int = 2,
                      verification_repeats: int = 3) -> IdentificationResult:
    """Identify a candidate map or reject an incomplete candidate set.

    Discovery queries use only the first N-validation_inputs states. Reserved
    states are queried verification_repeats times each in a held-out phase.
    The null hypothesis predicts uniform outputs. This reduces, but does not
    eliminate, false acceptance; it is not a calibrated universal novelty test.
    """
    unique = list(dict.fromkeys(tuple(c) for c in candidates))
    if not unique or len({len(c) for c in unique}) != 1:
        raise ValueError("candidates must be nonempty maps of equal size")
    n = len(unique[0])
    if len(true_map) != n or any(not 0 <= y < n for y in true_map):
        raise ValueError("true map size or output range differs from candidates")
    if any(any(not 0 <= y < n for y in c) for c in unique):
        raise ValueError("candidate output out of range")
    if not 0 < unknown_prior < 1 or not 0 < unknown_threshold < 1:
        raise ValueError("unknown prior and threshold must be in (0,1)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    if not 0 <= noise < 1 or strategy not in {"active", "fixed", "random"}:
        raise ValueError("invalid noise or strategy")
    if n < 2 or not 0 <= validation_inputs < n or verification_repeats < 1:
        raise ValueError("invalid validation configuration")
    if min_queries < 1 or max_queries < 1:
        raise ValueError("query budgets must be positive")
    rng = np.random.default_rng(seed)
    candidate_mass = np.full(len(unique), (1.0 - unknown_prior) / len(unique), dtype=float)
    p_unknown = float(unknown_prior)
    history = []
    decision = "uncertain"
    selection_n = n - validation_inputs

    def observe_and_update(x: int, phase: str):
        nonlocal candidate_mass, p_unknown
        y = int(true_map[x])
        if noise > 0 and rng.random() < noise:
            y = int(rng.choice([v for v in range(n) if v != y]))
        likelihood = np.asarray([
            (1.0 - noise) if c[x] == y else noise / (n - 1)
            for c in unique
        ], dtype=float)
        weighted = candidate_mass * likelihood
        unknown_weight = p_unknown / n
        total = float(weighted.sum() + unknown_weight)
        if total <= 0:
            return
        candidate_mass = weighted / total
        p_unknown = unknown_weight / total
        history.append({"phase": phase, "x": x, "y": y,
                        "best_candidate_posterior": float(candidate_mass.max()),
                        "unknown_posterior": p_unknown})

    for step in range(min(max_queries, min_queries)):
        if p_unknown >= unknown_threshold:
            decision = "rejected_unknown"
            break
        if strategy == "active":
            scores = [_open_set_information(unique, candidate_mass, p_unknown, x, noise)
                      for x in range(selection_n)]
            x = int(np.argmax(scores))
            if scores[x] <= 1e-12:
                break
        elif strategy == "fixed":
            x = step % selection_n
        else:
            x = int(rng.integers(0, selection_n))
        observe_and_update(x, "discovery")

    if decision != "rejected_unknown" and validation_inputs:
        for x in range(selection_n, n):
            for _ in range(verification_repeats):
                observe_and_update(x, "validation")
                if p_unknown >= unknown_threshold:
                    decision = "rejected_unknown"
                    break
            if decision == "rejected_unknown":
                break

    selected_index = int(np.argmax(candidate_mass))
    if decision != "rejected_unknown":
        if p_unknown >= unknown_threshold:
            decision = "rejected_unknown"
        elif candidate_mass[selected_index] >= confidence and p_unknown <= (1.0 - confidence):
            decision = "identified"
    identified = decision == "identified"
    return IdentificationResult(
        strategy=strategy, identified=identified,
        correct=identified and unique[selected_index] == tuple(true_map),
        queries=len(history), posterior_confidence=float(candidate_mass[selected_index]),
        selected_index=selected_index, history=history, decision=decision,
        unknown_probability=float(p_unknown))

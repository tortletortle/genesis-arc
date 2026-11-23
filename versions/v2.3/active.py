"""Information-gain active experiments for distinguishing finite-map hypotheses."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from typing import Sequence

Map = tuple[int, ...]

def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0.0)

def _channel_entropy(n_outputs: int, noise: float) -> float:
    if noise <= 0:
        return 0.0
    if noise >= 1:
        raise ValueError("noise must be < 1")
    probs = [1.0 - noise] + [noise / (n_outputs - 1)] * (n_outputs - 1)
    return _entropy(probs)

def query_information(candidates: Sequence[Map], posterior: Sequence[float],
                      x: int, noise: float = 0.0) -> float:
    """Mutual information I(H;Y|query=x) for symmetric output noise."""
    if not candidates or len(candidates) != len(posterior):
        raise ValueError("candidate and posterior lengths must match and be nonempty")
    n = len(candidates[0])
    if n < 2 or any(len(c) != n or any(not 0 <= y < n for y in c) for c in candidates):
        raise ValueError("candidate maps must have equal length and in-range outputs")
    if not 0 <= x < n or not 0 <= noise < 1:
        raise ValueError("invalid query or noise")
    p = np.asarray(posterior, dtype=float)
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("posterior must be nonnegative and sum to one")
    predicted = np.zeros(n, dtype=float)
    for weight, candidate in zip(p, candidates):
        predicted[candidate[x]] += weight
    observed = np.full(n, noise / (n - 1), dtype=float)
    observed += (1.0 - noise - noise / (n - 1)) * predicted
    return max(0.0, _entropy(observed) - _channel_entropy(n, noise))

def select_query(candidates: Sequence[Map], posterior: Sequence[float],
                 noise: float = 0.0, tolerance: float = 1e-12) -> int | None:
    """Choose maximum expected information gain; ties use the lowest input."""
    if not candidates:
        raise ValueError("candidate set must be nonempty")
    n = len(candidates[0])
    scores = [query_information(candidates, posterior, x, noise) for x in range(n)]
    best = max(scores)
    if best <= tolerance:
        return None
    return next(x for x, score in enumerate(scores) if abs(score - best) <= tolerance)

def update_posterior(candidates: Sequence[Map], posterior: Sequence[float],
                     x: int, observed_y: int, noise: float) -> list[float]:
    n = len(candidates[0])
    if not 0 <= x < n or not 0 <= observed_y < n or not 0 <= noise < 1:
        raise ValueError("invalid query, output, or noise")
    p = np.asarray(posterior, dtype=float)
    likelihood = np.asarray([
        (1.0 - noise) if candidate[x] == observed_y else noise / (n - 1)
        for candidate in candidates
    ], dtype=float)
    weighted = p * likelihood
    total = float(weighted.sum())
    if total <= 0:
        return [1.0 / len(candidates)] * len(candidates)
    return (weighted / total).tolist()

@dataclass
class IdentificationResult:
    strategy: str
    identified: bool
    correct: bool
    queries: int
    posterior_confidence: float
    selected_index: int
    history: list[dict]
    decision: str = "identified"
    unknown_probability: float = 0.0

def identify(candidates: Sequence[Map], true_map: Map, noise: float = 0.0,
             confidence: float = 0.95, max_queries: int = 100,
             strategy: str = "active", seed: int = 0) -> IdentificationResult:
    """Closed-set identification using active, fixed-cycle, or random queries."""
    unique = list(dict.fromkeys(tuple(c) for c in candidates))
    if not unique or len({len(c) for c in unique}) != 1:
        raise ValueError("candidates must be nonempty maps of equal size")
    if tuple(true_map) not in unique:
        raise ValueError("true map must be in candidate set")
    if strategy not in {"active", "fixed", "random"}:
        raise ValueError("unknown strategy")
    rng = np.random.default_rng(seed)
    posterior = [1.0 / len(unique)] * len(unique)
    history = []
    n = len(unique[0])
    for step in range(max_queries):
        top = int(np.argmax(posterior))
        if posterior[top] >= confidence:
            return IdentificationResult(strategy, True, unique[top] == tuple(true_map),
                                        step, float(posterior[top]), top, history)
        if strategy == "active":
            x = select_query(unique, posterior, noise)
            if x is None:
                break
        elif strategy == "fixed":
            x = step % n
        else:
            x = int(rng.integers(0, n))
        y = int(true_map[x])
        if noise > 0 and rng.random() < noise:
            y = int(rng.choice([v for v in range(n) if v != y]))
        posterior = update_posterior(unique, posterior, x, y, noise)
        history.append({"x": x, "y": y, "confidence": max(posterior),
                        "information_bits": query_information(unique, posterior, x, noise)})
    top = int(np.argmax(posterior))
    return IdentificationResult(strategy, False, unique[top] == tuple(true_map),
                                len(history), float(posterior[top]), top, history)

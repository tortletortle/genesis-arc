"""Seeded algebraic latent-factor identifiability experiment."""
import numpy as np
from latent import assess_identifiability, gauge_transform, pairwise_composites, recover_canonical_gauge

def random_permutation(rng, n=8, fix_zero=False):
    if fix_zero:
        return (0, *tuple(map(int, rng.permutation(np.arange(1, n)))))
    return tuple(map(int, rng.permutation(n)))

def run_factorization(seeds: int):
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(230000 + seed)
        primitives = [random_permutation(rng) for _ in range(4)]
        composites = pairwise_composites(primitives)
        anchored = assess_identifiability(composites, len(primitives), (0, primitives[0]))
        unanchored = assess_identifiability(composites, len(primitives))
        h = random_permutation(rng)
        transformed = gauge_transform(primitives, h)
        gauge_invariant = pairwise_composites(primitives) == pairwise_composites(transformed)
        canonical = recover_canonical_gauge(composites, len(primitives))
        rows.append({
            "seed": seed,
            "pairwise_relations": len(composites),
            "anchored_exact_recovery": anchored.unique_original_primitives_identifiable and anchored.canonical_model == tuple(primitives),
            "anchored_model_consistent": anchored.pairwise_consistent,
            "unanchored_canonical_consistent": unanchored.pairwise_consistent,
            "unanchored_original_unique": unanchored.unique_original_primitives_identifiable,
            "gauge_equivalence_lower_bound": unanchored.gauge_equivalence_class_size_lower_bound,
            "gauge_invariant": gauge_invariant,
            "canonical_model_matches_original": tuple(canonical) == tuple(primitives),
        })
    return rows

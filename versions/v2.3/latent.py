"""Latent-operator recovery and identifiability for finite permutation maps.

Composition convention matches V2.2: compose(a, b) returns b after a,
so compose(a, b)[x] == b[a[x]].
"""
from __future__ import annotations
from dataclasses import dataclass
from math import factorial
from typing import Mapping, Sequence

Map = tuple[int, ...]

def validate_permutation(p: Sequence[int]) -> Map:
    out = tuple(map(int, p))
    if sorted(out) != list(range(len(out))):
        raise ValueError("map must be a permutation of 0..N-1")
    return out

def identity(n: int) -> Map:
    if n < 1:
        raise ValueError("state count must be positive")
    return tuple(range(n))

def compose(a: Map, b: Map) -> Map:
    """Return b o a."""
    if len(a) != len(b):
        raise ValueError("map sizes differ")
    return tuple(b[a[x]] for x in range(len(a)))

def inverse(p: Map) -> Map:
    validate_permutation(p)
    inv = [0] * len(p)
    for x, y in enumerate(p):
        inv[y] = x
    return tuple(inv)

def recover_second_factor(composite: Map, first_factor: Map) -> Map:
    """Given C = B o A and A, recover B = C o A^-1."""
    if len(composite) != len(first_factor):
        raise ValueError("map sizes differ")
    return compose(inverse(first_factor), composite)

def recover_first_factor(composite: Map, second_factor: Map) -> Map:
    """Given C = B o A and B, recover A = B^-1 o C."""
    if len(composite) != len(second_factor):
        raise ValueError("map sizes differ")
    return compose(composite, inverse(second_factor))

def pairwise_composites(primitives: Sequence[Map]) -> dict[tuple[int, int], Map]:
    """C[i,j] = P_j o P_i^-1 for every ordered pair i != j."""
    ps = [validate_permutation(p) for p in primitives]
    if not ps or len({len(p) for p in ps}) != 1:
        raise ValueError("primitives must be nonempty and have equal map sizes")
    return {(i, j): compose(inverse(ps[i]), ps[j])
            for i in range(len(ps)) for j in range(len(ps)) if i != j}

def recover_from_anchor(composites: Mapping[tuple[int, int], Map],
                        anchor_index: int, anchor: Map, count: int) -> list[Map]:
    """Recover P_j from C[anchor,j] = P_j o P_anchor^-1 and known P_anchor."""
    if count < 2 or not 0 <= anchor_index < count:
        raise ValueError("invalid anchor index or operator count")
    anchor = validate_permutation(anchor)
    recovered: list[Map | None] = [None] * count
    recovered[anchor_index] = anchor
    for j in range(count):
        if j == anchor_index:
            continue
        key = (anchor_index, j)
        if key not in composites:
            raise KeyError(f"missing anchored composite {key}")
        recovered[j] = compose(anchor, composites[key])
    return [p for p in recovered if p is not None]

def recover_canonical_gauge(composites: Mapping[tuple[int, int], Map],
                            count: int, reference_index: int = 0) -> list[Map]:
    """Recover a canonical relative basis with Q_ref = identity.

    The Q_i are identifiable from pairwise composites, but generally are not
    the original P_i. They represent the equivalence class under shared
    right-composition (gauge) transforms.
    """
    if count < 2 or not 0 <= reference_index < count:
        raise ValueError("invalid reference index or operator count")
    sample = next(iter(composites.values()), None)
    if sample is None:
        raise ValueError("at least one pairwise composite is required")
    n = len(sample)
    for value in composites.values():
        validate_permutation(value)
        if len(value) != n:
            raise ValueError("pairwise composite sizes differ")
    q: list[Map | None] = [None] * count
    q[reference_index] = identity(n)
    for j in range(count):
        if j == reference_index:
            continue
        key = (reference_index, j)
        if key not in composites:
            raise KeyError(f"missing reference composite {key}")
        q[j] = validate_permutation(composites[key])
    return [p for p in q if p is not None]

def verify_pairwise_model(composites: Mapping[tuple[int, int], Map],
                          operators: Sequence[Map], tolerance: int = 0) -> dict:
    """Check all observed pairwise relations against a proposed operator set."""
    errors = []
    checked = 0
    for (i, j), observed in composites.items():
        if not (0 <= i < len(operators) and 0 <= j < len(operators)):
            raise ValueError("pairwise relation index out of range")
        observed = validate_permutation(observed)
        predicted = compose(inverse(operators[i]), operators[j])
        if len(predicted) != len(observed):
            raise ValueError("observed and predicted map sizes differ")
        distance = sum(a != b for a, b in zip(predicted, observed))
        checked += 1
        if distance > tolerance:
            errors.append({"pair": (i, j), "hamming": distance})
    return {"checked": checked, "consistent": not errors, "errors": errors}

def gauge_transform(primitives: Sequence[Map], h: Map) -> list[Map]:
    """Apply P_i' = P_i o H; pairwise composites remain invariant."""
    h = validate_permutation(h)
    return [compose(h, p) for p in primitives]

@dataclass(frozen=True)
class IdentifiabilityResult:
    status: str
    gauge_equivalence_class_size_lower_bound: int
    unique_original_primitives_identifiable: bool
    canonical_model: tuple[Map, ...]
    pairwise_consistent: bool

def assess_identifiability(composites: Mapping[tuple[int, int], Map], count: int,
                           anchor: tuple[int, Map] | None = None) -> IdentifiabilityResult:
    if count < 2:
        raise ValueError("at least two latent operators are required")
    expected = {(i, j) for i in range(count) for j in range(count) if i != j}
    if set(composites) != expected:
        raise ValueError("identifiability assessment requires the complete directed pairwise relation set")
    if anchor is None:
        canonical = tuple(recover_canonical_gauge(composites, count))
        check = verify_pairwise_model(composites, canonical)
        n = len(canonical[0])
        return IdentifiabilityResult(
            status="relative_basis_only_gauge_ambiguous",
            gauge_equivalence_class_size_lower_bound=factorial(n),
            unique_original_primitives_identifiable=False,
            canonical_model=canonical,
            pairwise_consistent=check["consistent"],
        )
    idx, primitive = anchor
    recovered = tuple(recover_from_anchor(composites, idx, primitive, count))
    check = verify_pairwise_model(composites, recovered)
    return IdentifiabilityResult(
        status="anchored_primitives_identified" if check["consistent"] else "anchor_model_rejected",
        gauge_equivalence_class_size_lower_bound=1 if check["consistent"] else factorial(len(primitive)),
        unique_original_primitives_identifiable=check["consistent"],
        canonical_model=recovered,
        pairwise_consistent=check["consistent"],
    )

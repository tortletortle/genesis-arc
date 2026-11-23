"""V2.2 self-organizing dictionary induction for finite deterministic maps.

The dictionary is induced only from observed skill maps. It has no access to
latent primitive labels or held-out maps. Programs are exact compositions of
selected observed maps; unencodable maps fall back to raw tables.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import ceil, log2
from typing import Iterable

N = 8
RAW_BITS = N * 3
META_BITS = 16
DICT_ENTRY_BITS = 8
LENGTH_BITS = 2
MAX_DEPTH = 3
MAX_DICT = 8

Map = tuple[int, ...]
Program = tuple[int, ...]

def compose(a: Map, b: Map) -> Map:
    """Return b after a: (b o a)(x) = b[a[x]]."""
    return tuple(b[a[x]] for x in range(N))

def apply_program(dictionary: list[Map] | tuple[Map, ...], program: Program) -> Map:
    if not program:
        raise ValueError('empty programs are not allowed')
    out = tuple(range(N))
    for token in program:
        out = compose(out, dictionary[token])
    return out

def token_bits(k: int) -> int:
    if k <= 0:
        raise ValueError('dictionary must contain at least one entry')
    return max(1, ceil(log2(k)))

def all_programs(dictionary: list[Map] | tuple[Map, ...], max_depth: int = MAX_DEPTH) -> dict[Map, Program]:
    """Enumerate shortest exact programs up to max_depth, stable tie-break."""
    result: dict[Map, Program] = {}
    if not dictionary:
        return result
    for depth in range(1, max_depth + 1):
        for seq in product(range(len(dictionary)), repeat=depth):
            key = apply_program(dictionary, seq)
            if key not in result:
                result[key] = tuple(seq)
    return result

def encode_one(key: Map, dictionary: list[Map] | tuple[Map, ...], programs=None):
    if key in dictionary:
        return ('atom', None, RAW_BITS)
    if programs is None:
        programs = all_programs(dictionary)
    seq = programs.get(key)
    if seq is None:
        return ('raw', None, RAW_BITS)
    return ('program', seq, LENGTH_BITS + len(seq) * token_bits(len(dictionary)))

def dataset_cost(keys: Iterable[Map], dictionary: list[Map] | tuple[Map, ...],
                 dict_entry_bits: int = DICT_ENTRY_BITS, metadata_bits: int = META_BITS,
                 max_depth: int = MAX_DEPTH) -> tuple[int, dict[Map, tuple[str, Program | None, int]]]:
    keys = list(dict.fromkeys(tuple(k) for k in keys))
    programs = all_programs(dictionary, max_depth)
    enc = {k: encode_one(k, dictionary, programs) for k in keys}
    bits = metadata_bits * len(keys) + dict_entry_bits * len(dictionary)
    bits += sum(v[2] for v in enc.values())
    return bits, enc

@dataclass
class FitResult:
    dictionary: list[Map]
    bits: int
    encodings: dict[Map, tuple[str, Program | None, int]]
    history: list[dict]

def fit_dictionary(keys: Iterable[Map], max_dict: int = MAX_DICT,
                    max_depth: int = MAX_DEPTH, pair_lookahead: bool = True) -> FitResult:
    """Greedy dictionary induction with pair look-ahead then marginal additions.

    Pair look-ahead avoids a zero-gain dead end: two independent primitives may
    jointly explain many compositions even when either alone explains none.
    Candidate atoms must be maps already present in the observed training set.
    """
    keys = sorted(set(tuple(k) for k in keys))
    if not keys:
        return FitResult([], 0, {}, [])
    dictionary: list[Map] = []
    history: list[dict] = []
    current, _ = dataset_cost(keys, dictionary, max_depth=max_depth)
    if pair_lookahead and max_dict >= 2 and len(keys) >= 2:
        best = None
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                cand = [keys[i], keys[j]]
                cost, _ = dataset_cost(keys, cand, max_depth=max_depth)
                gain = current - cost
                if best is None or gain > best[0] or (gain == best[0] and tuple(cand) < best[1]):
                    best = (gain, tuple(cand), cost)
        if best is not None and best[0] > 0:
            dictionary = list(best[1])
            current = best[2]
            history.append({'added': 2, 'gain_bits': best[0], 'total_bits': current,
                            'dict_size': len(dictionary), 'stage': 'pair_seed'})
    while len(dictionary) < max_dict:
        best = None
        for candidate in keys:
            if candidate in dictionary:
                continue
            cand = dictionary + [candidate]
            cost, _ = dataset_cost(keys, cand, max_depth=max_depth)
            gain = current - cost
            if best is None or gain > best[0] or (gain == best[0] and candidate < best[1]):
                best = (gain, candidate, cost)
        if best is None or best[0] <= 0:
            break
        dictionary.append(best[1])
        current = best[2]
        history.append({'added': 1, 'gain_bits': best[0], 'total_bits': current,
                        'dict_size': len(dictionary), 'stage': 'greedy'})
    bits, encodings = dataset_cost(keys, dictionary, max_depth=max_depth)
    return FitResult(dictionary, bits, encodings, history)

def decode_encoding(kind: str, payload: Program | None, key: Map,
                     dictionary: list[Map] | tuple[Map, ...]) -> Map:
    if kind in ('raw', 'atom'):
        return tuple(key)
    if kind != 'program' or payload is None:
        raise ValueError(f'invalid encoding: {kind}, {payload}')
    return apply_program(dictionary, payload)

def fixed_dictionary_cost(keys: Iterable[Map], dictionary: list[Map],
                          codebook_bits_per_entry: int = RAW_BITS + DICT_ENTRY_BITS,
                          max_depth: int = MAX_DEPTH):
    """Cost a preinstalled fixed DSL, charging its entire codebook once."""
    keys = list(dict.fromkeys(tuple(k) for k in keys))
    programs = all_programs(dictionary, max_depth)
    enc = {}
    for k in keys:
        seq = programs.get(k)
        if seq is None:
            enc[k] = ('raw', None, RAW_BITS)
        else:
            enc[k] = ('program', seq, LENGTH_BITS + len(seq) * token_bits(len(dictionary)))
    bits = META_BITS * len(keys) + codebook_bits_per_entry * len(dictionary)
    bits += sum(x[2] for x in enc.values())
    return bits, enc

def roundtrip(keys: Iterable[Map], dictionary: list[Map], encodings):
    decoded = {}
    for key in keys:
        kind, payload, _ = encodings[key]
        decoded[key] = decode_encoding(kind, payload, key, dictionary)
    return decoded

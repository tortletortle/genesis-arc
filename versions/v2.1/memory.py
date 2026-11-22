"""Lossless skill-memory codecs and bit-budgeted retention for Genesis ARC V2.1."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import v19_baseline as b

META_BITS = 16
RAW_MAP_BITS = b.N * 3
PRIMITIVE_BITS = 4
LENGTH_BITS = 2  # current grammar has programs of length 1..3
CHANGE_COUNT_BITS = 4
INDEX_BITS = 3
VALUE_BITS = 3

@dataclass
class Record(b.Skill):
    pass

def program_for(key):
    return b.v.PROGRAMS.get(tuple(key))

def program_bits(key):
    p = program_for(key)
    return None if p is None else LENGTH_BITS + PRIMITIVE_BITS * len(p)

def decode_program(program):
    x = b.X.copy()
    for name in program:
        x = b.v.PRIMS[name][x]
    return tuple(map(int, x))

def hamming(a, c):
    return sum(int(x != y) for x, y in zip(a, c))

def _mst_parent(keys):
    """Minimum spanning tree with a virtual root; returns parent index per map."""
    n = len(keys)
    if n == 0:
        return [], 0
    ptr_bits = math.ceil(math.log2(n)) if n > 1 else 0
    root_cost = [min(program_bits(k) or RAW_MAP_BITS, RAW_MAP_BITS) for k in keys]
    total = n + 1
    inf = 10**9
    best = [inf] * total
    parent = [-1] * total
    used = [False] * total
    best[n] = 0
    for _ in range(total):
        u = min((i for i in range(total) if not used[i]), key=lambda i: best[i])
        used[u] = True
        for v in range(total):
            if used[v] or v == u:
                continue
            if u == n:
                w = root_cost[v]
            elif v == n:
                w = root_cost[u]
            else:
                d = hamming(keys[u], keys[v])
                w = ptr_bits + CHANGE_COUNT_BITS + (INDEX_BITS + VALUE_BITS) * d
            if w < best[v]:
                best[v] = w
                parent[v] = u
    return parent[:n], int(sum(best[:n]))

class BitMemory:
    """Exact skill maps with raw, program, or lossless delta bit accounting."""
    def __init__(self, budget_bits: int, codec: str):
        assert codec in ('raw', 'program', 'delta')
        self.cap = int(budget_bits)
        self.codec = codec
        self.data: dict[tuple, Record] = {}
        self.evicted = 0

    def _encoded_records(self):
        keys = list(self.data)
        if not keys:
            return [], 0
        if self.codec == 'raw':
            records = [('raw', k, RAW_MAP_BITS) for k in keys]
            return records, len(keys) * RAW_MAP_BITS
        if self.codec == 'program':
            records = []
            bits = 0
            for k in keys:
                p = program_for(k)
                if p is None:
                    records.append(('raw', k, RAW_MAP_BITS)); bits += RAW_MAP_BITS
                else:
                    c = LENGTH_BITS + PRIMITIVE_BITS * len(p)
                    records.append(('program', k, tuple(p), c)); bits += c
            return records, bits
        parent, code_bits = _mst_parent(keys)
        records = []
        ptr_bits = math.ceil(math.log2(len(keys))) if len(keys) > 1 else 0
        children = [[] for _ in keys]
        roots = []
        for i, p in enumerate(parent):
            if p == len(keys): roots.append(i)
            else: children[p].append(i)
        stack = list(roots)
        while stack:
            i = stack.pop()
            p = parent[i]
            if p == len(keys):
                prog = program_for(keys[i])
                if prog is None:
                    records.append(('raw', i, keys[i], RAW_MAP_BITS))
                else:
                    c = LENGTH_BITS + PRIMITIVE_BITS * len(prog)
                    records.append(('program', i, tuple(prog), c))
            else:
                diffs = tuple((j, keys[i][j]) for j in range(b.N) if keys[i][j] != keys[p][j])
                c = ptr_bits + CHANGE_COUNT_BITS + (INDEX_BITS + VALUE_BITS) * len(diffs)
                records.append(('patch', i, p, diffs, c))
            stack.extend(children[i])
        return records, code_bits

    @property
    def bits_used(self):
        _, code = self._encoded_records()
        return len(self.data) * META_BITS + code

    def decode_all(self):
        keys = list(self.data)
        if self.codec == 'raw':
            return {k: tuple(k) for k in keys}
        records, _ = self._encoded_records()
        if self.codec == 'program':
            out = {}
            for rec in records:
                if rec[0] == 'raw':
                    _, k, _ = rec; out[k] = tuple(k)
                else:
                    _, k, prog, _ = rec; out[k] = decode_program(prog)
            return out
        out = {}
        pending = list(records)
        while pending:
            progress = False
            rest = []
            for rec in pending:
                kind = rec[0]
                if kind == 'raw':
                    _, i, k, _ = rec; out[i] = tuple(k); progress = True
                elif kind == 'program':
                    _, i, p, _ = rec; out[i] = decode_program(p); progress = True
                else:
                    _, i, p, diffs, _ = rec
                    if p not in out:
                        rest.append(rec); continue
                    k = list(out[p])
                    for j, value in diffs: k[j] = value
                    out[i] = tuple(k); progress = True
            if not progress and rest:
                raise ValueError('cyclic or unresolved delta references')
            pending = rest
        return {keys[i]: out[i] for i in range(len(keys))}

    def validate(self):
        decoded = self.decode_all()
        return decoded == {k: tuple(k) for k in self.data}

    def update(self, agent, episode):
        old_items = list(self.data.items())
        for i, (k, skill) in enumerate(old_items):
            q = max(agent.dist(o)[2][i] for o in range(b.O))
            skill.utility = .86 * skill.utility + .14 * q
            if q > .5:
                skill.last = episode
        for o in range(b.O):
            p, exact, _, _ = agent.dist(o)
            k = b.v.key(p.argmax(1))
            conf = float(np.mean(p[b.X, k]))
            coverage = np.count_nonzero(agent.counts[o].sum(1)) / b.N
            if conf < .65 or (coverage < .5 and exact[1] < .5):
                continue
            if k in self.data:
                self.data[k].conf = max(self.data[k].conf, conf)
            else:
                self.data[k] = Record(conf, episode, episode, .1 + .25 * conf)
        removed = 0
        while self.bits_used > self.cap and self.data:
            victim = min(self.data, key=lambda k: (
                self.data[k].utility + .08 * self.data[k].conf
                + .08 / (1 + episode - self.data[k].last),
                self.data[k].born, k))
            del self.data[victim]
            removed += 1
        self.evicted += removed
        if not self.validate():
            raise AssertionError('memory codec failed lossless round-trip')
        return removed

    def stats(self):
        keys = list(self.data)
        dsl = sum(k in b.IDX for k in keys)
        return {
            'skills': len(keys), 'bits_used': self.bits_used,
            'bits_per_skill': self.bits_used / len(keys) if keys else 0.,
            'dsl_skills': dsl, 'outside_dsl_skills': len(keys) - dsl,
            'evicted_total': self.evicted,
        }

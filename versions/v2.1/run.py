"""Genesis ARC V2.1 capacity sweep and lossless memory compression benchmark."""
import argparse, json, time
from pathlib import Path
import numpy as np
import v19_baseline as b
from memory import BitMemory

METHODS = {
    'raw_6': ('raw', 6 * 40),
    'raw_12': ('raw', 12 * 40),
    'raw_24': ('raw', 24 * 40),
    'raw_48': ('raw', 48 * 40),
    'program_480': ('program', 12 * 40),
    'delta_480': ('delta', 12 * 40),
    'program_960': ('program', 24 * 40),
    'delta_960': ('delta', 24 * 40),
}

def evaluate(mem, noise, world, tasks, uniforms, query_seed, budget=32):
    agent = b.Agent('utility_gate', noise, mem.data)
    rng = np.random.default_rng(query_seed)
    rows = []
    matches = sum(b.v.key(k) in mem.data for k in world)
    for q in range(budget + 1):
        if q in (0, 8, 16, 32):
            rows.append({
                'budget': q, 'bank_exact_matches': matches,
                'candidate_evals': agent.evals, 'gated': agent.gated,
                **mem.stats(), **b.score(agent, world, tasks),
            })
        if q < budget:
            op, x = agent.query(rng)
            agent.update(op, x, b.v.noisy(world[op, x], uniforms[q], noise))
    return agent, rows

def run(args):
    rows, memory_rows = [], []
    start = time.perf_counter()
    for seed in range(args.seeds):
        memories = {name: BitMemory(bits, codec) for name, (codec, bits) in METHODS.items()}
        frozen = None
        for episode, (phase, j, world, tasks, uniforms, qseed) in enumerate(
            b.stream(seed, args.offset, args.rounds)
        ):
            for name, mem in memories.items():
                agent, points = evaluate(mem, args.noise, world, tasks, uniforms, qseed)
                rows.extend(dict(seed=seed, phase=phase, episode=j, method=name, **p) for p in points)
                removed = mem.update(agent, episode)
                memory_rows.append(dict(seed=seed, phase=phase, episode=j, method=name,
                    removed=removed, **mem.stats()))
            if phase == 'acquire' and j == args.rounds - 1:
                r = np.random.default_rng(np.random.SeedSequence([221000 + args.offset, seed]))
                frozen = (world.copy(), b.v.tasks(r), r.random(16), int(r.integers(2**32)))
            if phase in ('acquire', 'shift') and j == args.rounds - 1 and frozen is not None:
                for name, mem in memories.items():
                    agent, points = evaluate(mem, args.noise, *frozen, budget=16)
                    rows.extend(dict(seed=seed, phase='probe_' + phase, episode=j, method=name, **p) for p in points if p['budget'] == 16)
        print('seed', seed, 'elapsed_s', round(time.perf_counter() - start, 1), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    b.v.write(args.out / 'runs.csv', rows)
    b.v.write(args.out / 'memory.csv', memory_rows)
    (args.out / 'metadata.json').write_text(json.dumps({
        'seeds': args.seeds, 'rounds': args.rounds, 'noise': args.noise,
        'offset': args.offset, 'methods': METHODS,
        'metadata_bits_per_skill': 16, 'raw_map_bits': 24,
        'primitive_bits': 4, 'program_length_bits': 2,
        'delta_pointer_bits': 'ceil(log2(number_of_skills))',
        'query_budget_per_environment': 32,
    }, indent=2))
    print('rows', len(rows), 'memory_rows', len(memory_rows), 'elapsed_s', round(time.perf_counter() - start, 1), flush=True)

def selftest():
    b.selftest()
    pm = BitMemory(100000, 'program')
    from memory import Record
    for i, k in enumerate(b.KEYS): pm.data[k] = Record(.9, i, i, .5)
    assert pm.validate()
    dm = BitMemory(100000, 'delta')
    out = []
    for i in range(b.N):
        k = tuple((x + i) % b.N for x in b.X)
        if k not in b.IDX and k not in out: out.append(k)
    sample = list(b.KEYS[:24]) + out
    for i, k in enumerate(sample): dm.data[k] = Record(.9, i, i, .5)
    assert dm.validate() and dm.bits_used > 0
    rng = np.random.default_rng(2101)
    for n in (1, 2, 3, 8, 16, 32, 48, 64):
        pool = list(b.KEYS)
        while len(pool) < len(b.KEYS) + 100:
            k = tuple(map(int, rng.integers(0, b.N, size=b.N)))
            if k not in b.IDX and k not in pool: pool.append(k)
        chosen = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
        probe = BitMemory(100000, 'delta')
        for i, k in enumerate(chosen): probe.data[k] = Record(.8, i, i, .4)
        assert probe.validate(), f'delta codec failed at n={n}'
        recs, cbits = probe._encoded_records()
        assert sum(r[-1] for r in recs) == cbits, f'bit accounting mismatch at n={n}'
    raw = BitMemory(100000, 'raw'); raw.data.update(dm.data)
    assert dm.bits_used <= raw.bits_used
    small = BitMemory(240, 'program')
    agent = b.Agent('utility_gate', .1, {})
    for x in b.X:
        for _ in range(3): agent.update(0, int(x), int((x + 1) % b.N))
    small.update(agent, 0)
    assert small.bits_used <= small.cap and small.validate()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=4)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--noise', type=float, default=.1)
    p.add_argument('--offset', type=int, default=40000)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    assert args.seeds > 0 and args.rounds > 0 and 0 < args.noise < .875
    selftest()
    run(args)

if __name__ == '__main__':
    main()

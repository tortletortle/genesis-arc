"""Controlled benchmark for self-organizing operator dictionaries (Genesis ARC V2.2)."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import v17_baseline as base
from dictionary import (N, RAW_BITS, META_BITS, DICT_ENTRY_BITS, MAX_DEPTH,
                        all_programs, dataset_cost, fit_dictionary,
                        fixed_dictionary_cost, roundtrip)

NOISES = (0.05, 0.20)
OBS_REPEATS = 5
TRAIN_COMPOSITES = 24
HELDOUT_COMPOSITES = 8
HIDDEN_PRIMITIVES = 4

def as_map(a):
    return tuple(map(int, a))

def random_hidden_primitives(rng, count=HIDDEN_PRIMITIVES):
    fixed = {as_map(p) for p in base.PRIMS.values()}
    fixed.add(tuple(range(N)))
    out = []
    while len(out) < count:
        p = as_map(rng.permutation(N))
        if p not in fixed and p not in out:
            out.append(p)
    return out

def compose_seq(prims, seq):
    out = tuple(range(N))
    for token in seq:
        out = tuple(prims[token][out[x]] for x in range(N))
    return out

def make_family(rng):
    prims = random_hidden_primitives(rng)
    seen = set(prims)
    train, held = [], []
    candidates = []
    for i in range(len(prims)):
        for j in range(len(prims)):
            if i != j:
                candidates.append((i, j))
    while len(candidates) < TRAIN_COMPOSITES + HELDOUT_COMPOSITES:
        length = int(rng.choice([2, 3, 3]))
        seq = tuple(map(int, rng.integers(0, len(prims), size=length)))
        if len(set(seq)) == 1:
            continue
        if seq not in candidates:
            candidates.append(seq)
    for seq in candidates:
        k = compose_seq(prims, seq)
        if k in seen:
            continue
        seen.add(k)
        (train if len(train) < TRAIN_COMPOSITES else held).append((seq, k))
        if len(train) == TRAIN_COMPOSITES and len(held) == HELDOUT_COMPOSITES:
            break
    if len(train) != TRAIN_COMPOSITES or len(held) != HELDOUT_COMPOSITES:
        raise RuntimeError('could not generate enough unique compositions')
    return prims, train, held

def observe_and_estimate(true_map, noise, repeats, rng):
    counts = np.zeros((N, N), dtype=np.int32)
    for x in range(N):
        for _ in range(repeats):
            u = float(rng.random())
            y = base.noisy(true_map[x], u, noise)
            counts[x, y] += 1
    estimate = []
    for x in range(N):
        best = np.flatnonzero(counts[x] == counts[x].max())
        estimate.append(int(rng.choice(best)))
    return tuple(estimate), counts

def method_costs(keys, hidden, fit):
    raw_bits = META_BITS * len(keys) + RAW_BITS * len(keys)
    dynamic_bits = fit.bits
    fixed = [as_map(p) for p in base.PRIMS.values()]
    fixed_bits, _ = fixed_dictionary_cost(keys, fixed)
    fixed_free_bits, _ = fixed_dictionary_cost(keys, fixed, codebook_bits_per_entry=0)
    oracle_bits, _ = dataset_cost(keys, hidden)
    no_pair = fit_dictionary(keys, pair_lookahead=False)
    rand_rng = np.random.default_rng(7741 + len(keys))
    random_costs = []
    if fit.dictionary:
        k = len(fit.dictionary)
        for _ in range(12):
            idx = rand_rng.choice(len(keys), size=min(k, len(keys)), replace=False)
            d = [keys[int(i)] for i in idx]
            random_costs.append(dataset_cost(keys, d)[0])
    random_bits = float(np.mean(random_costs)) if random_costs else raw_bits
    return {
        'raw_bits': raw_bits, 'dynamic_bits': dynamic_bits,
        'fixed_dsl_bits_charged': fixed_bits, 'fixed_dsl_bits_free_prior': fixed_free_bits,
        'oracle_bits': oracle_bits, 'no_pair_bits': no_pair.bits,
        'no_pair_dictionary_size': len(no_pair.dictionary),
        'random_dictionary_bits_mean': random_bits,
        'dynamic_saving_vs_raw_pct': 100 * (raw_bits - dynamic_bits) / raw_bits if raw_bits else 0,
        'dynamic_vs_fixed_free_prior_pct': 100 * (fixed_free_bits - dynamic_bits) / fixed_free_bits if fixed_free_bits else 0,
        'dynamic_vs_fixed_charged_pct': 100 * (fixed_bits - dynamic_bits) / fixed_bits if fixed_bits else 0,
    }

def fit_prefix(estimated, true_anchors, noise, seed, order_name):
    rows = []
    checkpoints = [4, 8, 12, 16, 20, 24, 28]
    for n in checkpoints:
        if n > len(estimated):
            continue
        keys = [x['estimate'] for x in estimated[:n]]
        unique = list(dict.fromkeys(keys))
        fit = fit_dictionary(unique)
        costs = method_costs(unique, true_anchors, fit)
        hidden = set(true_anchors)
        selected = set(fit.dictionary)
        precision = len(selected & hidden) / len(selected) if selected else 0.0
        recall = len(selected & hidden) / len(hidden) if hidden else 0.0
        rows.append(dict(seed=seed, noise=noise, order=order_name, prefix=n, unique_maps=len(unique),
                         dictionary_size=len(fit.dictionary), dictionary_precision=precision,
                         dictionary_recall=recall, **costs))
    return rows

def run_seed(seed, noise):
    rng = np.random.default_rng(np.random.SeedSequence([20261123, seed, int(noise * 1000)]))
    hidden, train, held = make_family(rng)
    all_tasks = [('anchor', (i,), k) for i, k in enumerate(hidden)] + [
        ('composite', seq, k) for seq, k in train
    ]
    estimates = []
    for kind, seq, true_map in all_tasks:
        estimate, counts = observe_and_estimate(true_map, noise, OBS_REPEATS, rng)
        estimates.append({'kind': kind, 'seq': seq, 'true': true_map,
                          'estimate': estimate, 'counts': counts})
    map_accuracy = float(np.mean([x['estimate'] == x['true'] for x in estimates]))
    anchor_order = list(range(len(estimates)))
    shuffle_order = list(map(int, rng.permutation(len(estimates))))
    prefix_rows = []
    for name, order in [('anchor_first', anchor_order), ('shuffled', shuffle_order)]:
        ordered = [estimates[i] for i in order]
        prefix_rows.extend(fit_prefix(ordered, hidden, noise, seed, name))
    train_est = [x['estimate'] for x in estimates]
    unique_train = list(dict.fromkeys(train_est))
    fit = fit_dictionary(unique_train)
    held_true = [k for _, k in held]
    held_est, held_correct = [], []
    for k in held_true:
        est, _ = observe_and_estimate(k, noise, OBS_REPEATS, rng)
        held_est.append(est); held_correct.append(est == k)
    _, held_enc = dataset_cost(held_est, fit.dictionary)
    held_decoded = roundtrip(held_est, fit.dictionary, held_enc)
    held_programs = sum(x[0] == 'program' for x in held_enc.values())
    held_bits = META_BITS * len(held_est) + sum(x[2] for x in held_enc.values()) + DICT_ENTRY_BITS * len(fit.dictionary)
    held_raw_bits = META_BITS * len(held_est) + RAW_BITS * len(held_est)

    novel_rng = np.random.default_rng(np.random.SeedSequence([99001, seed, int(noise * 1000)]))
    novel_hidden = random_hidden_primitives(novel_rng)
    visible = novel_hidden[:3]
    novel_train, novel_hold = [], []
    seen = set(novel_hidden)
    while len(novel_train) < 18 or len(novel_hold) < 6:
        length = int(novel_rng.choice([2, 3]))
        seq = tuple(map(int, novel_rng.integers(0, 4, size=length)))
        if 3 not in seq or len(set(seq)) == 1:
            continue
        k = compose_seq(novel_hidden, seq)
        if k in seen:
            continue
        seen.add(k)
        (novel_train if len(novel_train) < 18 else novel_hold).append((seq, k))
    visible_tasks = [(k, observe_and_estimate(k, noise, OBS_REPEATS, novel_rng)[0]) for k in visible]
    novel_train_est = [observe_and_estimate(k, noise, OBS_REPEATS, novel_rng)[0] for _, k in novel_train]
    novel_fit_keys = [e for _, e in visible_tasks] + novel_train_est
    novel_fit = fit_dictionary(list(dict.fromkeys(novel_fit_keys)))
    novel_hold_est = [observe_and_estimate(k, noise, OBS_REPEATS, novel_rng)[0] for _, k in novel_hold]
    _, novel_enc = dataset_cost(novel_hold_est, novel_fit.dictionary)
    novel_decoded = roundtrip(novel_hold_est, novel_fit.dictionary, novel_enc)
    novel_program_rate = sum(v[0] == 'program' for v in novel_enc.values()) / len(novel_enc)
    novel_dict_recall = len(set(novel_fit.dictionary) & set(novel_hidden)) / len(novel_hidden)
    novel_row = dict(seed=seed, noise=noise, visible_anchors=3,
                     visible_anchor_recall=len(set(novel_fit.dictionary) & set(visible)) / len(visible),
                     hidden_fourth_selected=int(novel_hidden[3] in novel_fit.dictionary),
                     train_composites=len(novel_train), heldout_composites=len(novel_hold),
                     dictionary_size=len(novel_fit.dictionary),
                     hidden_primitive_recall=novel_dict_recall,
                     heldout_program_rate=novel_program_rate,
                     heldout_roundtrip_rate=sum(novel_decoded[k] == k for k in novel_hold_est) / len(novel_hold_est),
                     heldout_map_accuracy=float(np.mean([e == k for e, (_, k) in zip(novel_hold_est, novel_hold)])))
    final_cost = method_costs(unique_train, hidden, fit)
    final_row = dict(seed=seed, noise=noise, map_accuracy=map_accuracy,
                     train_maps=len(unique_train), dictionary_size=len(fit.dictionary),
                     dictionary_precision=len(set(fit.dictionary) & set(hidden)) / max(1, len(fit.dictionary)),
                     dictionary_recall=len(set(fit.dictionary) & set(hidden)) / len(hidden),
                     program_rate=sum(v[0] == 'program' for v in fit.encodings.values()) / len(fit.encodings),
                     roundtrip_rate=sum(roundtrip(unique_train, fit.dictionary, fit.encodings)[k] == k for k in unique_train) / len(unique_train),
                     heldout_map_accuracy=float(np.mean(held_correct)),
                     heldout_program_rate=held_programs / len(held_est),
                     heldout_roundtrip_rate=sum(held_decoded[k] == k for k in held_est) / len(held_est),
                     heldout_bits=held_bits, heldout_raw_bits=held_raw_bits,
                     **final_cost)
    return prefix_rows, final_row, novel_row

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    prefixes, finals, novels = [], [], []
    start = time.perf_counter()
    for noise in args.noises:
        for seed in range(args.seeds):
            p, f, n = run_seed(seed + args.offset, noise)
            prefixes.extend(p); finals.append(f); novels.append(n)
            print('noise', noise, 'seed', seed + args.offset,
                  'elapsed_s', round(time.perf_counter() - start, 1), flush=True)
    write_csv(out / 'prefixes.csv', prefixes)
    write_csv(out / 'final.csv', finals)
    write_csv(out / 'novel_primitive.csv', novels)
    (out / 'metadata.json').write_text(json.dumps({
        'seeds_per_noise': args.seeds, 'seed_offset': args.offset,
        'noises': args.noises, 'observation_repeats_per_input': OBS_REPEATS,
        'hidden_primitives': HIDDEN_PRIMITIVES, 'train_composites': TRAIN_COMPOSITES,
        'heldout_composites': HELDOUT_COMPOSITES, 'max_dictionary_size': 8,
        'max_program_depth': MAX_DEPTH, 'raw_map_bits': RAW_BITS,
        'metadata_bits_per_skill': META_BITS,
        'dictionary_entry_overhead_bits': DICT_ENTRY_BITS,
        'novel_primitive_condition': '3 of 4 hidden primitives shown as anchors; held-out composites use all 4',
        'caveat': 'synthetic finite-map compression benchmark; not general intelligence'
    }, indent=2))
    print('prefix_rows', len(prefixes), 'final_rows', len(finals), 'novel_rows', len(novels), flush=True)

def selftest():
    rng = np.random.default_rng(22)
    d = [tuple(map(int, rng.permutation(N))) for _ in range(3)]
    for _ in range(20):
        seq = tuple(map(int, rng.integers(0, len(d), size=int(rng.integers(1, 4)))))
        k = d[seq[0]]
        for t in seq[1:]:
            k = tuple(d[t][k[x]] for x in range(N))
        assert k == __import__('dictionary').apply_program(d, seq)
    keys = list(dict.fromkeys([*d, *[__import__('dictionary').apply_program(d, (i, j)) for i in range(3) for j in range(3)]]))
    fit = fit_dictionary(keys)
    decoded = roundtrip(keys, fit.dictionary, fit.encodings)
    assert decoded == {k: k for k in keys}
    assert fit.bits <= META_BITS * len(keys) + RAW_BITS * len(keys)
    assert len(fit_dictionary(keys, pair_lookahead=False).dictionary) == 0
    held = tuple(map(int, rng.permutation(N)))
    assert held not in keys
    before = fit.bits
    _ = dataset_cost([held], fit.dictionary)
    assert fit.bits == before

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=16)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--noises', type=float, nargs='+', default=list(NOISES))
    p.add_argument('--out', required=True)
    args = p.parse_args()
    assert args.seeds > 0 and all(0 <= n < 0.875 for n in args.noises)
    selftest(); run(args)

if __name__ == '__main__':
    main()

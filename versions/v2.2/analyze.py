"""Summarize V2.2 dictionary experiments and paired seed-level bootstrap CIs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def mean_ci(values, rng, draws=10000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x): return (float('nan'), float('nan'), float('nan'))
    boot = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))

def main():
    p = argparse.ArgumentParser(); p.add_argument('folders', nargs='+'); args = p.parse_args()
    for folder in map(Path, args.folders):
        f = pd.read_csv(folder / 'final.csv')
        n = pd.read_csv(folder / 'novel_primitive.csv')
        pr = pd.read_csv(folder / 'prefixes.csv')
        rng = np.random.default_rng(20261123)
        cols = ['map_accuracy','dictionary_size','dictionary_precision','dictionary_recall',
                'program_rate','roundtrip_rate','dynamic_saving_vs_raw_pct',
                'dynamic_vs_fixed_free_prior_pct','dynamic_vs_fixed_charged_pct',
                'heldout_map_accuracy','heldout_program_rate']
        summary = f.groupby('noise')[cols].agg(['mean','std'])
        summary.to_csv(folder / 'summary.csv')
        effect_rows = []
        for noise, g in f.groupby('noise'):
            for label, values in [
                ('bits_saved_vs_raw', g.raw_bits - g.dynamic_bits),
                ('bits_saved_vs_fixed_free_prior', g.fixed_dsl_bits_free_prior - g.dynamic_bits),
                ('bits_saved_vs_random_dictionary', g.random_dictionary_bits_mean - g.dynamic_bits),
                ('excess_bits_vs_oracle', g.dynamic_bits - g.oracle_bits),
                ('dictionary_precision', g.dictionary_precision),
                ('heldout_program_rate', g.heldout_program_rate),
            ]:
                m, lo, hi = mean_ci(values, rng)
                effect_rows.append(dict(noise=noise, metric=label, n_seeds=len(g), mean=m, ci_low=lo, ci_high=hi))
        pd.DataFrame(effect_rows).to_csv(folder / 'paired_effects.csv', index=False)
        novel_cols = ['dictionary_size','hidden_primitive_recall','visible_anchor_recall',
                      'hidden_fourth_selected','heldout_program_rate','heldout_roundtrip_rate',
                      'heldout_map_accuracy']
        n.groupby('noise')[novel_cols].agg(['mean','std']).to_csv(folder / 'novel_summary.csv')
        prefix_summary = pr.groupby(['noise','order','prefix'])[['dictionary_size',
            'dictionary_precision','dictionary_recall','dynamic_saving_vs_raw_pct']].agg(['mean','std'])
        prefix_summary.to_csv(folder / 'prefix_summary.csv')
        discovery = []
        for (noise, order), g in pr.groupby(['noise','order']):
            for seed, sg in g.groupby('seed'):
                eligible = sg[(sg.dictionary_precision >= .75) & (sg.dynamic_saving_vs_raw_pct >= 5)]
                first = int(eligible.prefix.min()) if len(eligible) else -1
                discovery.append(dict(noise=noise, order=order, seed=seed, first_discovery_prefix=first))
        pd.DataFrame(discovery).to_csv(folder / 'discovery.csv', index=False)
        print('\n', folder, 'final means ± SD')
        print(f.groupby('noise')[cols].agg(['mean','std']).round(4).to_string())
        print('\nPaired bootstrap 95% CIs')
        print(pd.DataFrame(effect_rows).round(4).to_string(index=False))
        print('\nNovel hidden primitive')
        print(n.groupby('noise')[novel_cols].agg(['mean','std']).round(4).to_string())
        print('\nFirst useful dictionary prefix')
        print(pd.DataFrame(discovery).groupby(['noise','order']).first_discovery_prefix.agg(['mean','std']).round(3).to_string())

if __name__ == '__main__': main()

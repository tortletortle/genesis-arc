"""Summaries and paired seed-level bootstrap for V2.1 runs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PAIRS = [('program_480','raw_12'),('delta_480','raw_12'),('delta_480','program_480'),
         ('program_960','raw_24'),('delta_960','raw_24'),('delta_960','program_960'),
         ('raw_24','raw_12'),('raw_48','raw_24')]

def main():
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');a=p.parse_args()
    rng=np.random.default_rng(20261123)
    for folder in map(Path,a.folders):
        d=pd.read_csv(folder/'runs.csv')
        out=d.groupby(['phase','budget','method'])[['depth4','depth8','atomic','skills','bits_used','bits_per_skill','bank_exact_matches','candidate_evals']].mean().reset_index()
        out.to_csv(folder/'summary.csv',index=False)
        rows=[]
        # Pair within seed and phase, first return and frozen recall are primary endpoints.
        for phase in ('return','probe_shift','novel'):
            for budget in (16,32):
                z=d[(d.phase==phase)&(d.budget==budget)].groupby(['seed','method']).depth4.mean().unstack()
                for method,base in PAIRS:
                    if method not in z or base not in z: continue
                    diff=(z[method]-z[base]).dropna().to_numpy()
                    if not len(diff): continue
                    boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)
                    rows.append(dict(phase=phase,budget=budget,method=method,baseline=base,n_seeds=len(diff),delta=diff.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
        pd.DataFrame(rows).to_csv(folder/'paired_effects.csv',index=False)
        print('\n',folder,'rows',len(d),'seeds',d.seed.nunique())
        print(out[(out.budget==16)&out.phase.isin(['return','probe_shift','novel'])].pivot(index='method',columns='phase',values='depth4').round(4).to_string())
        print('\nmean retained skills / bits')
        print(out[(out.budget==16)&out.phase.eq('return')][['method','skills','bits_used','bits_per_skill']].round(2).to_string(index=False))

if __name__=='__main__':main()
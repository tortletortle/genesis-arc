import sys
from pathlib import Path
import numpy as np,pandas as pd
for folder in map(Path,sys.argv[1:] or ['results','robustness']):
 d=pd.read_csv(folder/'runs.csv'); m=pd.read_csv(folder/'memory.csv');
 d.groupby(['phase','budget','method'])[['atomic','depth4','depth8','bank_size','bank_exact_matches','candidate_evals','gated','calls']].mean().to_csv(folder/'summary.csv')
 m.groupby(['phase','method'])[['size','evicted','seconds','exact_dsl','outside_dsl']].mean().to_csv(folder/'memory_summary.csv')
 rng=np.random.default_rng(2026);effects=[]
 for (phase,budget),g in d.groupby(['phase','budget']):
  mat=g.groupby(['seed','method']).depth4.mean().unstack();
  for a,b in [('fifo_exact','no_memory'),('fifo_dual','fifo_exact'),('utility_dual','fifo_dual'),('utility_gate','utility_dual'),('utility_gate','no_memory'),('utility_gate','lookup')]:
   diff=(mat[a]-mat[b]).to_numpy();boot=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1);effects.append(dict(phase=phase,budget=budget,method=a,baseline=b,n_seeds=len(diff),delta=diff.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975)))
 pd.DataFrame(effects).to_csv(folder/'paired_effects.csv',index=False)
 print(folder, 'rows',len(d),len(m));print(d[d.budget==16].groupby(['method','phase']).depth4.mean().unstack().round(4));print(d[d.budget==32].groupby('method')[['candidate_evals','gated','calls','depth4']].mean().round(4));print(m.groupby('method')[['evicted','seconds','outside_dsl']].mean().round(4))

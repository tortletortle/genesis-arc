"""Merge independent V2.1 run folders while assigning disjoint seed IDs."""
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('inputs',nargs='+',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);runs=[];memory=[];offset=0;meta=[]
    for folder in a.inputs:
        cfg=json.loads((folder/'metadata.json').read_text());n=int(cfg['seeds'])
        r=pd.read_csv(folder/'runs.csv');m=pd.read_csv(folder/'memory.csv');r['seed']+=offset;m['seed']+=offset
        runs.append(r);memory.append(m);offset+=n;meta.append(cfg)
    pd.concat(runs,ignore_index=True).to_csv(a.out/'runs.csv',index=False)
    pd.concat(memory,ignore_index=True).to_csv(a.out/'memory.csv',index=False)
    result=dict(meta[0]);result['seeds']=offset;result['offsets']=[x.get('offset') for x in meta];result['merged_from']=[str(x) for x in a.inputs]
    (a.out/'metadata.json').write_text(json.dumps(result,indent=2));print('merged seeds',offset,'rows',sum(len(x) for x in runs))
if __name__=='__main__':main()

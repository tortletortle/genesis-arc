"""V2.0: evidence-based consolidation and frozen recall probes."""
import argparse, json, math
from pathlib import Path
from dataclasses import dataclass
import v19_baseline as b
import numpy as np

@dataclass
class Trace(b.Skill):
    support: int = 0
    credit: float = 0.

def predictive_credit(counts, key, noise):
    """Log evidence ratio: fixed remembered map versus independent uniform latent map."""
    L=counts*np.log(1-noise)+(counts.sum(1,keepdims=True)-counts)*np.log(noise/(b.N-1))
    independent=sum(b.v.logsumexp(row)-np.log(b.N) for row in L)
    return float(L[b.X,list(key)].sum()-independent)

class Consolidated(b.Memory):
    def __init__(self, cap, protect=True):
        super().__init__(cap,'utility_dual'); self.protect=protect
    def update(self,a,ep):
        # Only pre-existing skills earn independent reuse credit. No oracle or probe feedback.
        for k,s in self.data.items():
            eligible=[predictive_credit(c,k,a.noise) for c in a.counts if np.count_nonzero(c.sum(1))>=4]
            best=max(eligible,default=-math.inf)
            if best>=math.log(10):
                s.support+=1; s.credit+=min(best,8.); s.last=ep
        for o in range(b.O):
            p,exact,_,_=a.dist(o); k=b.v.key(p.argmax(1)); c=float(np.mean(p[b.X,k])); cover=np.count_nonzero(a.counts[o].sum(1))/b.N
            if c<.65 or (cover<.5 and exact[1]<.5): continue
            if k in self.data: self.data[k].conf=max(c,self.data[k].conf)
            else: self.data[k]=Trace(c,ep,ep,0.)
        stable=sorted((k for k,s in self.data.items() if s.support>0),key=lambda k:(self.data[k].support,self.data[k].credit,self.data[k].conf,k),reverse=True)[:self.cap//2] if self.protect else []
        removed=0
        while len(self.data)>self.cap:
            k=min((k for k in self.data if k not in stable),key=lambda k:(self.data[k].born,self.data[k].conf,k))
            del self.data[k]; removed+=1
        return removed

MODES=('fifo_exact','utility_gate','utility_2x','consolidated','unprotected')
def make(mode,cap):
    if mode in ('consolidated','unprotected'): return Consolidated(cap,mode=='consolidated')
    return b.Memory(cap*2 if mode=='utility_2x' else cap,'utility_gate' if mode=='utility_2x' else mode)
def evaluate(mem, mode, noise, world, ts, us, sr, budget=32):
    a=b.Agent('utility_gate' if mode!='fifo_exact' else mode,noise,mem.data); r=np.random.default_rng(sr)
    rows=[]; matches=sum(b.v.key(k) in mem.data for k in world)
    for q in range(budget+1):
        if q in (0,8,16,32): rows.append(dict(budget=q,bank_size=len(mem.data),bank_exact_matches=matches,**b.score(a,world,ts)))
        if q<budget:
            o,x=a.query(r);a.update(o,x,b.v.noisy(world[o,x],us[q],noise))
    return a,rows

def experiment(seeds,rounds,noise,offset,cap):
    rows=[]
    for seed in range(seeds):
        memories={m:make(m,cap) for m in MODES}; probe=None
        for ep,(phase,j,world,ts,us,sr) in enumerate(b.stream(seed,offset,rounds)):
            for m,mem in memories.items():
                a,rr=evaluate(mem,m,noise,world,ts,us,sr)
                rows.extend(dict(seed=seed,phase=phase,episode=j,method=m,**x) for x in rr)
                mem.update(a,ep); assert len(mem.data)<=mem.cap
            if phase=='acquire' and j==rounds-1:
                rng=np.random.default_rng(np.random.SeedSequence([220000+offset,seed]))
                probe=(world.copy(),b.v.tasks(rng),rng.random(32),int(rng.integers(2**32)))
            if (phase=='acquire' or phase=='shift') and j==rounds-1:
                for m,mem in memories.items():
                    _,rr=evaluate(mem,m,noise,*probe,budget=16)
                    rows.extend(dict(seed=seed,phase='probe_'+phase,episode=j,method=m,**x) for x in rr)
    return rows

def selftest():
    b.selftest(); k=b.KEYS[0]; c=np.zeros((b.N,b.N),int)
    for x in b.X:c[x,k[x]]=2
    assert predictive_credit(c,k,.1)>math.log(10)
    wrong=tuple((x+1)%b.N for x in k);assert predictive_credit(c,wrong,.1)<0
    mem=Consolidated(4);mem.data[k]=Trace(.95,0,0,0.)
    a=b.Agent('utility_gate',.1,mem.data);a.counts[0]=c.copy();mem.update(a,1);assert mem.data[k].support==1
    assert len(mem.data)<=4
    empty=np.zeros_like(c);assert abs(predictive_credit(empty,k,.1))<1e-10

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=8);p.add_argument('--rounds',type=int,default=6);p.add_argument('--noise',type=float,default=.1);p.add_argument('--offset',type=int,default=20000);p.add_argument('--capacity',type=int,default=12);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.seeds>0 and a.rounds>0 and a.capacity>1 and 0<a.noise<.875;selftest();a.out.mkdir(parents=True,exist_ok=True);rows=experiment(a.seeds,a.rounds,a.noise,a.offset,a.capacity);b.v.write(a.out/'runs.csv',rows);(a.out/'config.json').write_text(json.dumps({**vars(a),'out':str(a.out),'methods':MODES},indent=2));print(len(rows),'rows')
if __name__=='__main__':main()

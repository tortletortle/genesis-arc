"""ARC 1.7: bounded program synthesis, active falsification and audited skill transfer."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import argparse,csv,json,math
from pathlib import Path
N=8; O=4; X=np.arange(N); B=(0,8,16,24,32); MODES=('old_active','lookup_active','static_active','grow_passive','grow_active')
PRIMS={
 'add1':(X+1)%N,'xor1':X^1,'xor3':X^3,'reverse':N-1-X,
 'rotbits':((X<<1)&7)|(X>>2),'add3':(X+3)%N,
 'floorhalf':X//2,'satplus2':np.minimum(X+2,7),'threshold':np.where(X>=4,7,0),
 'clear_low':X&6,'double_mod':(2*X)%N,'swap_halves':X^4}
OLD=list(PRIMS)[:6]
def key(p):return tuple(map(int,p))
def grammar():
    seen={key(PRIMS[n]):(n,) for n in OLD}; tiers=[list(seen)]
    for d in range(1,4):
        new={}
        if d==1:
            for n,p in PRIMS.items():
                if key(p) not in seen:new[key(p)]=(n,)
        else:
            # extend every shortest known prefix by one primitive
            for k,prog in list(seen.items()):
                if len(prog)!=d-1:continue
                for n,p in PRIMS.items():
                    q=key(p[np.array(k)])
                    if q not in seen and q not in new:new[q]=prog+(n,)
        seen.update(new);tiers.append(list(new))
    return [np.array(t,dtype=int) for t in tiers],seen
TIERS,PROGRAMS=grammar(); WEIGHTS=np.array([.48,.12,.16,.16,.08]); ALL=np.concatenate(TIERS)

def logsumexp(x):
    m=x.max();return m+np.log(np.exp(x-m).sum())
def posterior(tables,counts,noise):
    # each row in counts contains observed outcomes at that input; exact likelihood
    match=counts[np.arange(N)[None,:],tables].sum(axis=1)
    logl=match*np.log(1-noise)+(counts.sum()-match)*np.log(noise/(N-1))
    z=logsumexp(logl); p=np.exp(logl-z)
    marg=np.stack([np.bincount(tables[:,x],p,minlength=N) for x in X])
    return z-np.log(len(tables)),marg,p

def lookup(counts,noise):
    logl=counts*np.log(1-noise)+(counts.sum(axis=1,keepdims=True)-counts)*np.log(noise/(N-1))
    m=logl.max(axis=1,keepdims=True);q=np.exp(logl-m);z=q.sum(axis=1,keepdims=True)
    return float((m+np.log(z)-np.log(N)).sum()),q/z

class Agent:
    def __init__(self,mode,noise,bank=(),threshold=1.):
        self.mode=mode;self.noise=noise;self.threshold=threshold
        self.tiers=[np.array(sorted(set(map(key,TIERS[0]))|set(bank))),*TIERS[1:]]
        # avoid double counting promoted maps across tiers
        bank_keys=set(map(key,self.tiers[0]))
        for i in range(1,4): self.tiers[i]=np.array([p for p in self.tiers[i] if key(p) not in bank_keys],dtype=int).reshape(-1,N)
        self.counts=np.zeros((O,N,N),int);self.stage=np.full(O,3 if mode=='static_active' else 0);self.cache=[None]*O;self.expansions=0;self.candidate_evals=0
    def distribution(self,o):
        if self.cache[o] is not None:return self.cache[o]
        c=self.counts[o];ez,lp=lookup(c,self.noise)
        if self.mode=='lookup_active':result=(lp,lp.argmax(axis=1),0.,None,0.)
        else:
            # Expand only when current program mixture is falsified relative to independent table model.
            while True:
                active=[i for i in range(self.stage[o]+1) if len(self.tiers[i])];parts=[posterior(self.tiers[i],c,self.noise) for i in active]
                self.candidate_evals+=sum(len(self.tiers[i]) for i in active)
                w=WEIGHTS[active];logs=np.array([a[0] for a in parts])+np.log(w)
                if self.mode.startswith('grow') and self.stage[o]<3 and c.sum()>=2 and ez-(logsumexp(logs)-np.log(w.sum()))>self.threshold:
                    self.stage[o]+=1;self.expansions+=1;continue
                break
            if self.mode=='old_active':
                weights=np.exp(logs-logsumexp(logs));p=parts[0][1]; probs=parts[0][2];ix=probs.argmax();candidate=key(self.tiers[0][ix]);conf=probs[ix];result=(p,self.tiers[0][ix],float(conf),candidate,0.)
            else:
                # Inactive prior mass is assigned to independent-table escape model. On expansion replay all evidence.
                logw=np.r_[logs,ez+np.log(1-w.sum())];mix=np.exp(logw-logsumexp(logw));p=sum(m * a[1] for m,a in zip(mix,parts))+mix[-1]*lp
                candidates=[(mix[j]*a[2].max(),key(self.tiers[i][a[2].argmax()])) for j,(i,a) in enumerate(zip(active,parts))]
                conf,candidate=max(candidates);result=(p,p.argmax(axis=1),float(conf),candidate,float(mix[-1]))
        self.cache[o]=result;return result
    def query(self,rng):
        if self.mode=='grow_passive':return tuple(rng.integers([O,N]))
        p=np.stack([self.distribution(o)[0] for o in range(O)]);y=p*(1-self.noise)+(1-p)*self.noise/(N-1)
        h=-(y*np.log(np.maximum(y,1e-20))).sum(axis=2);ix=np.argwhere(np.isclose(h,h.max(),atol=1e-10,rtol=0));return tuple(ix[rng.integers(len(ix))])
    def update(self,o,x,y):self.counts[o,x,y]+=1;self.cache[o]=None
    def predict(self):return np.stack([self.distribution(o)[1] for o in range(O)])

def noisy(y,u,n):return int(y if u>=n else (y+1+int((N-1)*u/n))%N)
def sample(pool,k,rng):return pool[rng.choice(len(pool),k,replace=False)]
def world(split,rng):
    if split=='familiar':return sample(TIERS[0],O,rng)
    if split=='novel_d3':return sample(TIERS[3],O,rng)
    if split=='mixed':return np.concatenate((sample(TIERS[0],2,rng),sample(TIERS[3],2,rng)))[rng.permutation(O)]
    out=[]
    while len(out)<O:
        q=rng.integers(N,size=N)
        if key(q) not in PROGRAMS and key(q) not in list(map(key,out)):out.append(q)
    return np.array(out)

def tasks(rng):return [(rng.integers(N,size=128),rng.integers(O,size=(128,d))) for d in (4,8)]
def score(agent,true,ts):
    p=agent.predict();un=agent.counts.sum(axis=2)==0;ret={'atomic':np.mean(p==true),'unqueried':np.mean((p==true)[un]) if un.any() else np.nan,'unqueried_n':int(un.sum()),'expansions':agent.expansions,'candidate_evals':agent.candidate_evals,'fallback_mass':np.mean([agent.distribution(o)[4] for o in range(O)])}
    for (x,ops),d in zip(ts,(4,8)):
        a=x.copy();b=x.copy()
        for o in ops.T:a=p[o,a];b=true[o,b]
        ret[f'depth{d}']=np.mean(a==b)
    return ret

def verify(agent,true,rng,noise):
    # independent verification AFTER discovery, exactly two fresh observations per proposed skill;
    # prefer untouched input values; never update/post-select using true map.
    bank=set();rows=[]
    for o in range(O):
        _,_,confidence,candidate,_=agent.distribution(o)
        if candidate is None or candidate in set(map(key,TIERS[0])) or confidence<.55:continue
        untouched=np.where(agent.counts[o].sum(axis=1)==0)[0];order=np.r_[rng.permutation(untouched),rng.permutation(np.setdiff1d(X,untouched))];xs=order[:2]
        ys=[noisy(true[o,x],u,noise) for x,u in zip(xs,rng.random(2))];passed=all(candidate[x]==y for x,y in zip(xs,ys));exact=key(true[o])==candidate
        if passed:bank.add(candidate)
        rows.append((o,confidence,passed,exact,int(np.isin(xs,untouched).sum()),candidate))
    return bank,rows

def write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run(args):
    rows=[];birth=[];transfer=[];n=args.noise
    for s in range(args.seeds):
        r=np.random.default_rng(170000+args.offset+s)
        for split in ['familiar','mixed','novel_d3','outside_dsl']:
            for c in range(args.contexts):
                true=world(split,r);ts=tasks(r);us=r.random(32);sr=int(r.integers(2**32));grown=None
                for mode in MODES:
                    a=Agent(mode,n);rng=np.random.default_rng(sr)
                    for b in range(33):
                        if b in B:rows.append(dict(seed=s,split=split,context=c,method=mode,budget=b,**score(a,true,ts)))
                        if b<32:
                            o,x=a.query(rng);a.update(o,x,noisy(true[o,x],us[b],n))
                    if mode=='grow_active':grown=a
                bank,vr=verify(grown,true,r,n)
                for o,conf,passed,exact,untouched,candidate in vr:birth.append(dict(seed=s,split=split,context=c,op=o,confidence=conf,passed=int(passed),exact=int(exact),untouched=untouched,program=" -> ".join(PROGRAMS[candidate]),table=" ".join(map(str,candidate))))
                # Independent target operation arrangements and query noise; same source family used for both arms.
                unverified={v[-1] for v in vr}
                for relation,target in [('same_family',true[r.permutation(O)]),('shifted_family',world('outside_dsl',r))]:
                    tts=tasks(r);tu=r.random(16);tr=int(r.integers(2**32))
                    for label,lib in [('no_birth',()),('verified_birth',bank),('unverified_birth',unverified)]:
                        a=Agent('grow_active',n,lib);rng=np.random.default_rng(tr)
                        for b in range(17):
                            if b in (0,8,16):transfer.append(dict(seed=s,split=split,context=c,relation=relation,method=label,budget=b,bank_size=len(lib),**score(a,target,tts)))
                            if b<16:
                                o,x=a.query(rng);a.update(o,x,noisy(target[o,x],tu[b],n))
    args.out.mkdir(parents=True,exist_ok=True);write(args.out/'runs.csv',rows);write(args.out/'birth.csv',birth);write(args.out/'transfer.csv',transfer)
    (args.out/'metadata.json').write_text(json.dumps(dict(seeds=args.seeds,contexts=args.contexts,seed_offset=args.offset,noise=n,tiers=[len(t) for t in TIERS],queries=32,verification_max_queries=8,transfer_queries=16),indent=2))
    print('tiers', [len(t) for t in TIERS], 'runs',len(rows),'birth',len(birth),'transfer',len(transfer),flush=True)

def selftest():
    assert all(len(t)>0 for t in TIERS);assert len(PROGRAMS)==len(ALL)==len(set(map(key,ALL)))
    for p,prog in PROGRAMS.items():
        v=X.copy()
        for n in prog:v=PRIMS[n][v]
        assert key(v)==p
    a=Agent('grow_active',.1);assert np.allclose(a.distribution(0)[0].sum(axis=1),1)
    for _ in range(4):a.update(0,0,2);a.distribution(0)
    assert a.stage[0]>0
    assert np.isclose(lookup(np.zeros((N,N)),.1)[0],0)
    assert np.isclose(posterior(TIERS[0],np.zeros((N,N)),.1)[0],0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=16);p.add_argument('--contexts',type=int,default=4);p.add_argument('--noise',type=float,default=.1);p.add_argument('--offset',type=int,default=0);p.add_argument('--out',type=Path,default=Path(__file__).parent/'results');a=p.parse_args();assert 0<a.noise<.875;selftest();run(a)
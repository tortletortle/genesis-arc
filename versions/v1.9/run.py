"""Continual skill retrieval and adaptive compute, fixed query budget."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import argparse,json,time
from pathlib import Path
from dataclasses import dataclass
import v17_baseline as v
N,O,X=v.N,v.O,v.X; T=v.ALL; K=len(T); KEYS=[v.key(t) for t in T]; IDX={k:i for i,k in enumerate(KEYS)}
BASE=np.concatenate([np.full(len(t),w/len(t)) for t,w in zip(v.TIERS,[.48,.12,.16,.16])]);BASE/=BASE.sum(); EYE=np.eye(N); ONE=EYE[T]; Q=.76*ONE+.24/N
MODES=['lookup','no_memory','fifo_exact','fifo_dual','utility_dual','utility_gate']
def norm(l):
 z=v.logsumexp(l);return z,np.exp(l-z)
@dataclass
class Skill:
 conf:float; born:int; last:int; utility:float
class Memory:
 def __init__(self,cap,mode):self.cap=cap;self.mode=mode;self.data={}
 def update(self,a,ep):
  if self.mode in ('lookup','no_memory'):return 0
  for i,(k,s) in enumerate(self.data.items()):
   q=max(a.dist(o)[2][i] for o in range(O));s.utility=.86*s.utility+.14*q
   if q>.5:s.last=ep
  for o in range(O):
   p,exact,_,_=a.dist(o);k,c=exact
   if self.mode=='fifo_exact':
    if k is None or c<.40:continue
   else:
    k=v.key(p.argmax(1));c=float(np.mean(p[X,k]));cover=np.count_nonzero(a.counts[o].sum(1))/N
    if c<.65 or (cover<.5 and exact[1]<.5):continue
   if k in self.data:
    s=self.data[k];s.conf=max(s.conf,c);s.last=ep
   else:self.data[k]=Skill(c,ep,ep,.1+.25*c)
  n=0
  while len(self.data)>self.cap:
   def value(k):
    s=self.data[k];return (s.born,k) if self.mode.startswith('fifo') else (s.utility+.08*s.conf+.08/(1+ep-s.last),s.born,k)
   del self.data[min(self.data,key=value)];n+=1
  return n
class Agent:
 def __init__(self,mode,noise,mem):
  self.mode=mode;self.noise=noise;self.counts=np.zeros((O,N,N),int);self.cache=[None]*O;self.evals=0;self.gated=0;self.calls=0
  self.keys=list(mem);self.bt=np.array(self.keys,int).reshape(-1,N);self.bo=EYE[self.bt];self.bq=.76*self.bo+.24/N;w=np.array([max(.05,s.conf) for s in mem.values()]);self.bw=w/w.sum() if len(w) else w
 def components(self,t,one,q,w,L,E,shift):
  self.evals+=2*len(t);l=L[X[None,:],t].sum(1);z,p=norm(l+np.log(w));m=np.einsum('k,kxy->xy',p,one)
  den=np.einsum('kxy,xy->kx',q,E);z2,p2=norm(np.log(den).sum(1)+shift.sum()+np.log(w));m2=np.einsum('k,kxy,kx,xy->xy',p2,q,1/den,E,optimize=True)
  return z,p,m,z2,p2,m2
 def dist(self,o):
  if self.cache[o] is not None:return self.cache[o]
  self.calls+=1;c=self.counts[o];L=c*np.log(1-self.noise)+(c.sum(1,keepdims=True)-c)*np.log(self.noise/(N-1));shift=L.max(1);E=np.exp(L-shift[:,None]);p=E/E.sum(1,keepdims=True);z=np.log(E.mean(1)).sum()+shift.sum()
  if self.mode=='lookup':self.cache[o]=(p,(None,0.),np.zeros(0),1.);return self.cache[o]
  logs=[z];marg=[p];weights=[.18];exact=[];bp=[]
  if len(self.keys):
   a,b,c,d,e,f=self.components(self.bt,self.bo,self.bq,self.bw,L,E,shift);logs += [a,d];marg += [c,f];weights += [.34,.16];exact.append((1,self.keys,b));bp=[(1,b),(2,e)]
  gate=False
  if self.mode=='utility_gate' and self.keys and self.counts[o].sum()>=4:
   _,small=norm(np.array(logs)+np.log(weights));gate=small[1]>.85 and (self.counts[o].sum(1)>0).sum()>=3
  if gate:self.gated+=1
  else:
   a,b,c,d,e,f=self.components(T,ONE,Q,BASE,L,E,shift);i=len(logs);logs += [a,d];marg += [c,f];weights += [.25,.07] if self.keys else [.64,.18];exact.append((i,KEYS,b))
  _,post=norm(np.array(logs)+np.log(weights));pred=sum(w*m for w,m in zip(post,marg));bank=sum((post[i]*q for i,q in bp),start=np.zeros(len(self.keys)))
  candidates={}
  for i,keys,b in exact:
   j=int(b.argmax());candidates[keys[j]]=0.
  for k in candidates:
   for i,keys,b in exact:
    j=IDX.get(k) if keys is KEYS else (keys.index(k) if k in keys else None)
    if j is not None:candidates[k]+=float(post[i]*b[j])
  k=max(candidates,key=candidates.get);self.cache[o]=(pred,(k,candidates[k]),bank,float(post[0]));return self.cache[o]
 def query(self,r):
  ps=[self.dist(o) for o in range(O)];p=np.stack([a[0] for a in ps]);y=p*(1-self.noise)+(1-p)*self.noise/(N-1);h=-(y*np.log(np.maximum(y,1e-30))).sum(2)+.65*(self.counts.sum(2)==0)*np.array([a[3] for a in ps])[:,None];ii=np.argwhere(np.isclose(h,h.max(),atol=1e-10,rtol=0));return tuple(ii[r.integers(len(ii))])
 def update(self,o,x,y):self.counts[o,x,y]+=1;self.cache[o]=None
 def predict(self):return np.stack([self.dist(o)[0].argmax(1) for o in range(O)])
def stream(seed,offset,rounds):
 r=np.random.default_rng(np.random.SeedSequence([190000+offset,seed]));exact=v.TIERS[3][r.choice(len(v.TIERS[3]),12,replace=False)];outside=[]
 while len(outside)<12:
  p=r.integers(N,size=N);k=v.key(p)
  if k not in IDX and k not in list(map(v.key,outside)):outside.append(p)
 outside=np.array(outside);pools=[(exact[:6],outside[:6]),(exact[6:],outside[6:])]
 for phase,group in [('acquire',0),('shift',1),('return',0),('novel',None)]:
  for j in range(rounds):
   if group is None:true=v.world('outside_dsl',r)
   else:
    a,b=pools[group];true=np.concatenate([v.sample(a,2,r),v.sample(b,2,r)])[r.permutation(O)]
   yield phase,j,true,v.tasks(r),r.random(32),int(r.integers(2**32))
def score(a,true,ts):
 p=a.predict();ret={'atomic':float(np.mean(p==true))}
 for (x,ops),d in zip(ts,(4,8)):
  g=x.copy();h=x.copy()
  for o in ops.T:g=p[o,g];h=true[o,h]
  ret[f'depth{d}']=float(np.mean(g==h))
 return ret
def run(args):
 rows=[];memrows=[];t=time.perf_counter()
 for s in range(args.seeds):
  memories={m:Memory(args.capacity,m) for m in MODES}
  for ep,(phase,j,true,ts,us,sr) in enumerate(stream(s,args.offset,args.rounds)):
   for m,mem in memories.items():
    a=Agent(m,args.noise,mem.data);r=np.random.default_rng(sr);start=time.perf_counter()
    for b in range(33):
     if b in (8,16,32):
      met=score(a,true,ts);rows.append(dict(seed=s,phase=phase,episode=j,method=m,budget=b,bank_size=len(a.keys),bank_exact_matches=sum(v.key(k) in mem.data for k in true),candidate_evals=a.evals,gated=a.gated,calls=a.calls,**met))
     if b<32:
      o,x=a.query(r);a.update(o,x,v.noisy(true[o,x],us[b],args.noise))
    ev=mem.update(a,ep);memrows.append(dict(seed=s,phase=phase,episode=j,method=m,size=len(mem.data),evicted=ev,seconds=time.perf_counter()-start,exact_dsl=sum(k in IDX for k in mem.data),outside_dsl=sum(k not in IDX for k in mem.data)))
  print('seed',s,'seconds',round(time.perf_counter()-t,1),flush=True)
 args.out.mkdir(parents=True,exist_ok=True);v.write(args.out/'runs.csv',rows);v.write(args.out/'memory.csv',memrows);(args.out/'metadata.json').write_text(json.dumps(dict(seeds=args.seeds,rounds=args.rounds,noise=args.noise,offset=args.offset,capacity=args.capacity,query_budget=32,methods=MODES),indent=2));print(len(rows),len(memrows),flush=True)
def selftest():
 v.selftest();a=Agent('no_memory',.1,{});assert np.allclose(a.dist(0)[0].sum(1),1)
 k=KEYS[0];a=Agent('utility_gate',.1,{k:Skill(.95,0,0,.5)});assert np.allclose(a.dist(0)[0].sum(1),1)
 for x in X:
  for _ in range(3):a.update(0,int(x),k[x])
 assert np.all(a.dist(0)[0].argmax(1)==k);assert a.gated>0
 mem=Memory(1,'utility_dual');mem.data[k]=Skill(.95,0,0,.5);mem.update(a,0);assert len(mem.data)<=1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=8);p.add_argument('--rounds',type=int,default=6);p.add_argument('--capacity',type=int,default=12);p.add_argument('--noise',type=float,default=.1);p.add_argument('--offset',type=int,default=0);p.add_argument('--out',type=Path,default=Path('results'));args=p.parse_args();assert 0<args.noise<.875 and args.capacity>0;selftest();run(args)

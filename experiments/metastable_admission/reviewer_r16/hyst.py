"""Reviewer check R-16 (not part of E-28): large-spike and long-horizon hysteresis test. M1, 380k."""
import sys,os,csv,itertools,statistics as st
sys.path.insert(0,'/Users/kushagrabainsls/Desktop/mscs/masters-thesis/experiments/metastable_admission')
import run_e28 as R
from sim import Sim
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
H=12000.0
MIX,CACHE='M1_all',380000
g,nstar=R.calibration()[('base',MIX,CACHE)]
mt=R.mean_turns(MIX)
LOADS=[0.73,0.85,1.0,1.1]   # offered turns/s (1.0 x knee is 0.73)
EXTRA=[0,1,2,4,8]           # extra sessions in 60 s, in units of N*
CL={'A':(60.0,2),'C':(600.0,2)}
def go(a):
    L,e,cl,seed=a
    rate=L/mt
    thin=1+max(EXTRA)*nstar/(rate*60)
    mult=1+e*nstar/(rate*60) if e else 1.0
    to,rt=CL[cl]
    p=R.params(MIX,CACHE,base_rate=rate,spike_mult=mult,thin_mult=thin,timeout_s=to,max_retries=rt,seed=seed,horizon_s=H,spike_len_s=60.0 if e else 0.0)
    specs,gaps=R.pool(MIX)
    bins=Sim(p,specs,gaps).run()
    def w(k,a0,a1): return sum(v[k] for t,v in bins.items() if a0<=t*10<a1)/(a1-a0)
    act=lambda t: bins.get(int(t//10),{}).get('active',-1)
    out=dict(L=L,e=e,cl=cl,seed=seed,pre=w('turn_ok',1800,2400),post=w('turn_ok',3060,3660),mid=w('turn_ok',6000,7000),late=w('turn_ok',10400,12000),
      a_pre=act(2390),a_3000=act(3000),a_6000=act(6000),a_9000=act(9000),a_end=act(11990),pre_pre=w('preempt',1800,2400),late_pre=w('preempt',10400,12000),late_to=w('timeouts',10400,12000),arr_late=w('arrivals',10400,12000))
    return out
if __name__=='__main__':
    jobs=[(L,e,cl,s) for L in LOADS for e in EXTRA for cl in CL for s in range(1,7)]
    with ProcessPoolExecutor(8) as ex: res=list(ex.map(go,jobs,chunksize=1))
    with open(os.path.dirname(os.path.abspath(__file__))+'/hyst.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print('done',len(res))

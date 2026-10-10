"""Reviewer check R-16 (not E-28): does a static cap clear the bad state? Also A-client session failures."""
import sys,os,csv,statistics as st
sys.path.insert(0,'/Users/kushagrabainsls/Desktop/mscs/masters-thesis/experiments/metastable_admission')
import run_e28 as R
from sim import Sim
from concurrent.futures import ProcessPoolExecutor
H=12000.0; MIX,CACHE='M1_all',380000
g,nstar=R.calibration()[('base',MIX,CACHE)]; mt=R.mean_turns(MIX)
CL={'A':(60.0,2),'C':(600.0,2)}
def go(a):
    L,e,cl,cap,seed=a
    rate=L/mt; thin=1+8*nstar/(rate*60); mult=1+e*nstar/(rate*60) if e else 1.0
    to,rt=CL[cl]
    kw=dict(policy='cap',session_cap=cap) if cap else {}
    p=R.params(MIX,CACHE,base_rate=rate,spike_mult=mult,thin_mult=thin,timeout_s=to,max_retries=rt,seed=seed,horizon_s=H,spike_len_s=60.0 if e else 0.0,**kw)
    specs,gaps=R.pool(MIX); bins=Sim(p,specs,gaps).run()
    w=lambda k,a0,a1: sum(v[k] for t,v in bins.items() if a0<=t*10<a1)/(a1-a0)
    return dict(L=L,e=e,cl=cl,cap=cap,seed=seed,late=w('turn_ok',10400,12000),late_pre=w('preempt',10400,12000),
      a_end=bins[1199]['active'],sessfail_all=sum(v['sess_fail'] for v in bins.values()),sessok_all=sum(v['sess_ok'] for v in bins.values()),
      sessfail_spike=sum(v['sess_fail'] for t,v in bins.items() if 2400<=t*10<4200),arrivals=sum(v['arrivals'] for v in bins.values()))
if __name__=='__main__':
    jobs=[(0.73,4,'C',cap,s) for cap in (0,12,16) for s in range(1,7)]
    jobs+=[(0.85,0,'C',cap,s) for cap in (12,16) for s in range(1,7)]
    jobs+=[(1.0,0,'C',cap,s) for cap in (12,16) for s in range(1,7)]
    jobs+=[(0.73,e,'A',0,s) for e in (0,4,8) for s in range(1,7)]
    with ProcessPoolExecutor(8) as ex: res=list(ex.map(go,jobs,chunksize=1))
    import collections
    gp=collections.defaultdict(list)
    for r in res: gp[(r['L'],r['e'],r['cl'],r['cap'])].append(r)
    for k,l in sorted(gp.items()):
        m=lambda x: st.mean(r[x] for r in l)
        print(k,'late %.2f preempt %.3f active_end %.0f sessfail %.1f/%.0f arrivals, fail in spike window %.1f'%(m('late'),m('late_pre'),m('a_end'),m('sessfail_all'),m('arrivals'),m('sessfail_spike')))

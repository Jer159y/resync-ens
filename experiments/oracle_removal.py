"""Exact bounded-truth counterexample: perfect failure identification need not improve a population.

No simulation noise: integrate against U[-1,1] analytically. Scores evaluate the
actual finite empirical forecast; no fair/independent-member correction.
"""
import csv,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/oracle_removal'
def quantile(xs,p):
    xs=sorted(xs);v=(len(xs)-1)*p;i=int(v);j=min(i+1,len(xs)-1)
    return xs[i]+(v-i)*(xs[j]-xs[i])
def w1_uniform(xs):
    xs=sorted(xs);n=len(xs);s=0.
    for i,x in enumerate(xs):
        c=(x+1)/2;lo=i/n-c;hi=(i+1)/n-c
        s+=hi*abs(hi)-lo*abs(lo)
    return s
def expected_abs(x):
    return abs(x) if abs(x)>=1 else (x*x+1)/2
def expected_crps(xs):
    xs=sorted(xs);n=len(xs)
    first=statistics.mean(expected_abs(x) for x in xs)
    dispersion=sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*n)
    return first-dispersion
def crps_at(xs,y):
    xs=sorted(xs);n=len(xs)
    return statistics.mean(abs(x-y) for x in xs)-sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*n)
def coverage(xs,level):
    lo=quantile(xs,(1-level)/2);hi=quantile(xs,(1+level)/2)
    return max(0.,min(1.,hi)-max(-1.,lo))/2
def metrics(xs):
    cov=coverage(xs,.9)
    return dict(n=len(xs),w1=w1_uniform(xs),expected_crps=expected_crps(xs),coverage90=cov,
                coverage90_abs_error=abs(cov-.9),spread=statistics.stdev(xs),
                mean_prediction_rmse=math.sqrt(1/3+statistics.mean(xs)**2),
                member_mae_at_truth0=statistics.mean(abs(x) for x in xs),
                ensemble_mean_abs_error_at_truth0=abs(statistics.mean(xs)),crps_at_truth0=crps_at(xs,0.))
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=[];cdfs=[]
    for case,remaining in [('balanced_remaining',[-1+(i+.5)*2/200 for i in range(200)]),
                           ('left_biased_remaining',[-1.]*200),
                           ('distinct_left_biased_remaining',[-1+(i+.5)*.2/200 for i in range(200)])]:
        original=remaining+[2.]*100
        # Oracle knows the true physical support; every selected member is impossible.
        bad=[i for i,x in enumerate(original) if not -1<=x<=1]
        kept=[x for i,x in enumerate(original) if i not in bad]
        assert bad==list(range(200,300)) and kept==remaining
        assert min(abs(original[i]) for i in bad)>max(abs(x) for x in kept)
        fixed_size=kept+[kept[0]]*100 if case=='left_biased_remaining' else None
        arms=[('before',original),('oracle_remove',kept)]
        if fixed_size is not None:arms.append(('remove_and_clone_keep_N300',fixed_size))
        for policy,xs in arms:
            results.append(dict(case=case,policy=policy,oracle_true_positives=len(bad),oracle_false_positives=0,
                                **metrics(xs)))
            for j in range(351):
                x=-1.2+.01*j
                cdfs.append(dict(case=case,policy=policy,x=x,empirical_cdf=sum(v<=x for v in xs)/len(xs),
                                 truth_cdf=max(0.,min(1.,(x+1)/2))))
    biased=next(r for r in results if r['case']=='left_biased_remaining' and r['policy']=='before')
    removed=next(r for r in results if r['case']=='left_biased_remaining' and r['policy']=='oracle_remove')
    assert math.isclose(biased['w1'],8/9,abs_tol=1e-11)
    assert math.isclose(biased['expected_crps'],2/3,abs_tol=1e-11)
    assert math.isclose(removed['w1'],1.,abs_tol=1e-11)
    assert math.isclose(removed['expected_crps'],1.,abs_tol=1e-11)
    assert removed['coverage90']==0. and biased['coverage90']==1.
    assert removed['member_mae_at_truth0']<biased['member_mae_at_truth0']
    assert removed['crps_at_truth0']>biased['crps_at_truth0']
    distinct=[r for r in results if r['case']=='distinct_left_biased_remaining']
    assert distinct[1]['w1']>distinct[0]['w1'] and distinct[1]['expected_crps']>distinct[0]['expected_crps']
    cloned=next(r for r in results if r['policy']=='remove_and_clone_keep_N300')
    for key in ['w1','expected_crps','coverage90']:assert math.isclose(cloned[key],removed[key],abs_tol=1e-11)
    balanced=[r for r in results if r['case']=='balanced_remaining']
    assert balanced[1]['w1']<balanced[0]['w1'] and balanced[1]['expected_crps']<balanced[0]['expected_crps']
    sweep=[]
    for j in range(141):
        outlier=1.01+.05*j;before=metrics([-1.]*200+[outlier]*100)
        sweep.append(dict(outlier=outlier,delta_w1_remove_minus_before=1-before['w1'],
                          delta_crps_remove_minus_before=1-before['expected_crps']))
    write(OUT/'results.csv',results);write(OUT/'cdfs.csv',cdfs);write(OUT/'outlier_sweep.csv',sweep)
    provenance=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        truth='U[-1,1], analytic population integrals; no sampled test truths',N_before=300,N_after_removal=200,
        oracle='remove exactly the 100 constant +2 members outside true support; zero false positives; also exactly worst point errors at matched truth y=0',
        retained_cases=['200 balanced in-support quantiles','200 identical -1 predictions: extreme shared bias','200 distinct quantiles in [-1,-0.8]: nondegenerate shared-bias robustness'],
        intervention='drop impossible members and renormalize remaining weights; no replay or model training',
        distinction='perfect identification of these impossible members does not make retained in-support members well-distributed',
        score='finite empirical CRPS with N^2 denominator; expected over independent U[-1,1] truth',
        checks=['W1 exact 8/9 to 1','expected CRPS exact 2/3 to 1','coverage 1 to 0','exact worst-error members selected at truth y=0; member MAE improves but CRPS worsens',
                'fixed-N clone control has same W1/CRPS/coverage as removal','balanced case improves W1/CRPS'],
        scope='constructed counterexample to a universal guarantee; not evidence of frequency or cause in ESN runs')
    (OUT/'protocol.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()

"""Reaggregate primary numerical results from frozen per-window CSVs, no simulations."""
from pathlib import Path
import csv,math,statistics,json
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/compact';OUT=ROOT/'build/tables';OUT.mkdir(parents=True,exist_ok=True)
def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def save(name,data):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
result=[];nchecks=0
for expected in rows(DATA/'main_cases_20261123/five_case_summary.csv'):
    id=expected['system'];row={'system':id,'k':int(expected['k']),'multiplier':int(expected['multiplier'])}
    for policy,k,m in [('baseline',0,1),('resync',row['k'],row['multiplier'])]:
        d=DATA/'main'/id/'test'/f'k{k}_m{m}'
        base=rows(d/'effective_base_windows.csv');dist=rows(d/'effective_distribution_windows.csv')
        assert len(base)==200 and len(dist)==20,(id,len(base),len(dist))
        for metric,values in [('w1',dist),('spread',base),('spread_loss',base),('crps',base)]:
            actual=statistics.mean(float(v[metric]) for v in values);recorded=float(expected[policy+'_'+metric])
            assert math.isclose(actual,recorded,rel_tol=1e-12,abs_tol=1e-12),(id,policy,metric,actual,recorded)
            row[policy+'_'+metric]=actual;nchecks+=1
    for metric in ['w1','spread','spread_loss','crps']:
        row[metric+'_change_percent']=100*(row['resync_'+metric]/row['baseline_'+metric]-1)
    result.append(row)
save('primary_results.csv',result)
# Event contrasts retain the shared-parent event structure, without pseudo-replication CI.
contrast=rows(DATA/'controls/paired_event_contrasts.csv');groups={}
for r in contrast:groups.setdefault((r['case'],r['metric'],r['contrast']),[]).append(float(r['paired_difference']))
def quantile(v,p):
    v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
save('paired_event_contrasts.csv',[{'system':k[0],'metric':k[1],'contrast':k[2],'n_dependent_events':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'q25':quantile(v,.25),'q75':quantile(v,.75)} for k,v in sorted(groups.items())])
history=rows(DATA/'followup_results_20261125/history.csv');rep=rows(DATA/'followup_results_20261125/replication.csv')
assert len(history)==56 and len(rep)==24,(len(history),len(rep))
save('history_length_results.csv',history);save('crossed_replication.csv',rep)
report={'primary_numeric_checks':nchecks,'systems':len(result),'history_policies':len(history),'crossed_policy_outcomes':len(rep),'independent_run_CI':False}
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

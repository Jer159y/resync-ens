"""Verified compact inputs for descriptive history and crossed-replication plots."""
from pathlib import Path
import csv, hashlib, json, tomllib

R=Path(__file__).resolve().parents[1];S=R/'data/compact/followup_results_20261125';S.mkdir(exist_ok=True)
sources={}
def read(p):
    sources[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open() as f:return list(csv.DictReader(f))
def toml(p):
    sources[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return tomllib.loads(p.read_text())
def save(name,rows):
    with (S/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
completed=R/'data/compact/ks_followup_completion_20261127'
audit=json.loads((completed/'remote_verification.json').read_text());sources[str((completed/'remote_verification.json').relative_to(R))]=hashlib.sha256((completed/'remote_verification.json').read_bytes()).hexdigest()
assert audit['passed'] and audit['policies']==28 and audit['own_and_returned_valid']
assert max(x['max_own_returned_window_metric_difference_from_full'] for x in audit['plateau_window_metric_comparison'])==0
old_dense=read(R/'data/compact/resync_history_dense_20261124/dense_summary.csv')
dense=read(completed/'dense_summary.csv')
dense=[x for x in dense if x['scope']=='returned']
for case in ['duffing','ks200']:
    q=[x for x in dense if x['case']==case]
    assert len(q)==28 and all(x['status']=='complete' and x['threads']=='2' for x in q)
assert [x for x in dense if x['case']=='duffing']==[x for x in old_dense if x['case']=='duffing' and x['scope']=='returned']
first=set()
for x in [x for x in dense if x['case']=='ks200']:
    folder=completed/'history/ks200'/x['policy'];c=toml(folder/'complete.toml');first.add(c['first_own_sha256'])
    assert c['status']=='complete' and c['base_intervals']==200
    a=read(folder/'returned_summary.csv')[0]
    assert a['valid']=='true' and a['excluded_missing_truth']==a['nonfinite_returned_values']=='0'
    for key in ['w1','spread_loss','crps']:assert float(a[key])==float(x[key])
    assert c['forecast_seconds']==float(x['forecast_seconds'])
assert len(first)==1
save('history.csv',dense)
rows=[]
old=R/'data/compact/revision_chaos_pre_20261124/replication_snapshot/duffing'
ext=R/'data/compact/resync_reserve_extension_20261125/snapshot'
for bank in [310000,320000,330000]:
    for physical in [2026112401,2026112402]:
        expanded=(bank,physical)==(330000,2026112402)
        folder=ext if expanded else old/f'bank{bank}_physical{physical}'
        cohort=toml(folder/'cohort.toml');assert cohort['converged'] and len(cohort['member_ids'])==300
        records={p:toml(folder/p/'complete.toml') for p in ['none','full']}
        assert all(x['status']=='complete' and x['base_intervals']==200 for x in records.values())
        assert records['none']['first_own_sha256']==records['full']['first_own_sha256']
        for policy in ['none','full']:
            q=read(folder/policy/'returned_summary.csv')[0]
            assert q['valid']=='true' and q['excluded_missing_truth']==q['nonfinite_returned_values']=='0'
            rows.append(dict(case='duffing',bank_seed_start=bank,physical_seed=physical,policy=policy,
                             w1=q['w1'],spread=q['spread'],spread_loss=q['spread_loss'],crps=q['crps'],
                             expanded_reserve=expanded,total_model_cap=750 if expanded else 450))
for bank in [310000,320000,330000]:
    for physical in [2026112401,2026112402]:
        folder=completed/'replication/ks200'/f'bank{bank}_physical{physical}'
        cohort=toml(folder/'cohort.toml');assert cohort['converged'] and len(set(cohort['member_ids']))==300 and max(cohort['member_ids'])<=450
        records={p:toml(folder/p/'complete.toml') for p in ['none','full']}
        assert all(x['status']=='complete' and x['base_intervals']==200 for x in records.values())
        assert records['none']['first_own_sha256']==records['full']['first_own_sha256']
        for policy in ['none','full']:
            q=read(folder/policy/'returned_summary.csv')[0]
            assert q['valid']=='true' and q['excluded_missing_truth']==q['nonfinite_returned_values']=='0'
            rows.append(dict(case='ks200',bank_seed_start=bank,physical_seed=physical,policy=policy,
                             w1=q['w1'],spread=q['spread'],spread_loss=q['spread_loss'],crps=q['crps'],expanded_reserve=False,total_model_cap=450))
save('replication.csv',rows)
ranges={}
for case in ['duffing','ks200']:
    values={k:[] for k in ['w1','spread_loss','crps']}
    for bank in [310000,320000,330000]:
        for physical in [2026112401,2026112402]:
            pair={x['policy']:x for x in rows if x['case']==case and x['bank_seed_start']==bank and x['physical_seed']==physical}
            for k in values:
                gain=100*(1-float(pair['full'][k])/float(pair['none'][k]));assert gain>0;values[k].append(gain)
    ranges[case]={k:dict(min=min(v),max=max(v)) for k,v in values.items()}
out=dict(source_sha256=sources,history_policies=56,history_policies_by_case=dict(duffing=28,ks200=28),replication_pairs=12,replication_pairs_by_case=dict(duffing=6,ks200=6),
         crossed_bank_levels=3,crossed_physical_levels=2,ks_status='complete_verified',improvement_percent_ranges=ranges,
         reserve_extension_pair=[330000,2026112402],independent_iid_CI=False,
         scope='descriptive postselection follow-ups; original primary tests unchanged')
(S/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['history_policies','replication_pairs','ks_status']}))

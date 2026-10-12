"""Resolve ua for the exact paired members used by the eight-model height test."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import pandas as pd
import requests
from build_cmip6_multimember_opendap_manifest import endpoint, years, REQUIRED_YEARS, API
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/manifests/cmip6_multimember_ua_revision_files.csv'
STATUS=ROOT/'data/processed/cmip6_multimember_ua_revision_resolution.json'

def resolve(row):
    query={'type':'File','project':'CMIP6','source_id':row['model'],
        'experiment_id':row['experiment'],'member_id':row['member'],
        'variable_id':'ua','table_id':'Amon','grid_label':row['grid'],
        'latest':'true','format':'application/solr+json','limit':1000}
    response=requests.get(API,params=query,timeout=(15,60));response.raise_for_status()
    payload=response.json()['response']
    if payload['numFound']>1000: raise RuntimeError('Search pagination required; refusing truncated catalogue')
    docs=[x for x in payload['docs'] if endpoint(x)]
    records=[]
    for doc in docs:
        filename=str(doc.get('title',''))
        start,end=years(filename)
        version=doc.get('version','')
        if isinstance(version,list): version=version[0]
        if start is None: continue
        records.append({**row,'variable':'ua','resolved_version':str(version),
            'filename':filename,'file_start_year':start,'file_end_year':end,
            'opendap_url':endpoint(doc),'dataset_id':str(doc.get('instance_id',''))})
    frame=pd.DataFrame(records)
    if frame.empty: raise RuntimeError('No ua OPeNDAP files resolved')
    # Do not mix versions to manufacture complete temporal coverage.
    for version in sorted(frame.resolved_version.unique(),reverse=True):
        group=frame.loc[frame.resolved_version.eq(version)].copy()
        group['replica_priority']=~group.opendap_url.str.contains('esgf-node.ornl.gov',regex=False)
        group=group.sort_values(['replica_priority','opendap_url']).drop_duplicates('filename')
        covered=set()
        for item in group.itertuples(): covered.update(range(item.file_start_year,item.file_end_year+1))
        if REQUIRED_YEARS.issubset(covered): return group.drop(columns='replica_priority').to_dict('records')
    raise RuntimeError('No single version covers both response periods')

def main():
    height=pd.read_csv(ROOT/'data/processed/cmip6_multimember_z850_responses.csv')
    columns=['model','member','experiment']
    groups=height[columns].drop_duplicates()
    source=pd.read_csv(ROOT/'data/processed/cmip6_multimember_zarr_validation.csv')
    work=groups.merge(source[columns+['grid']].drop_duplicates(),on=columns,validate='one_to_one')
    assert len(work)==48 and work.model.nunique()==8
    rows=[]; failures=[];status={'status':'resolving','expected_groups':48,'completed_groups':0,'failures':[]}
    def checkpoint():
        status['failures']=failures
        temporary=STATUS.with_suffix('.json.part');temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(STATUS)
        if rows: pd.DataFrame(rows).to_csv(OUT,index=False)
    checkpoint()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(resolve,row):row for row in work.to_dict('records')}
        for future in as_completed(futures):
            row=futures[future]
            try:
                found=future.result();rows.extend(found);status['completed_groups']+=1
                print(f"resolved {row}: {len(found)} files",flush=True)
            except Exception as exc:
                failures.append({**row,'error':f'{type(exc).__name__}: {exc}'})
                print(f"failed {row}: {exc}",flush=True)
            checkpoint()
    status['status']='resolved' if not failures else 'incomplete';checkpoint()
    if failures: raise SystemExit(2)

if __name__=='__main__':main()

"""Exploratory, catalogue-conditioned winter genesis exposure/rate decomposition."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed'
PREFIX = 'winter_genesis_regimes_'
SEED = 20261004
YEARS = np.arange(1980, 2024)


def canonical(ds):
    ds = ds.rename({a: b for a, b in [('valid_time','time'), ('latitude','lat'),
                                    ('longitude','lon'), ('pressure_level','level')]
                    if a in ds.coords})
    if 'level' in ds.dims:
        ds = ds.sel(level=700, drop=True)
    return ds.sortby('lat').sortby('lon')


def target_times():
    return pd.DatetimeIndex(np.concatenate([
        pd.date_range(f'{y-1}-12-01', f'{y}-03-01', freq='6h', inclusive='left').values
        for y in YEARS]))


def extract(times):
    env = times - pd.Timedelta(hours=24)
    lat = np.arange(24.,39.)
    lon = np.arange(100.,116.)
    pieces, pressures, stamps, sources = [], [], [], []
    for year in sorted(set(env.year)):
        wanted = env[env.year == year]
        path = ROOT / f'data/raw/era5_700hpa_6hourly/era5_700hpa_{year}.nc'
        spath = ROOT / f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc'
        with xr.open_dataset(path) as raw:
            ds = canonical(raw).sel(time=wanted, lat=lat, lon=lon)
            a = np.stack([ds[v].values for v in ['z','u','v']], axis=1)
        with xr.open_dataset(spath) as raw:
            p = canonical(raw).sp.sel(time=wanted, lat=lat, lon=lon).values
        pieces.append(a.reshape(len(wanted),3,-1))
        pressures.append(p.reshape(len(wanted),-1))
        stamps.extend(wanted)
        for source in [path,spath]:
            stat = source.stat()
            sources.append({'path':source.relative_to(ROOT).as_posix(),
                            'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        print(f'extract antecedent background {year}: {len(wanted)} times', flush=True)
    if not pd.DatetimeIndex(stamps).equals(env):
        raise RuntimeError('Environment timestamps mismatch')
    a = np.concatenate(pieces)
    p = np.concatenate(pressures)
    la, lo = np.meshgrid(lat,lon,indexing='ij')
    return a,p,la.ravel(),lo.ravel(),sources


def decompose(exposure, counts, early_ids, late_ids):
    e0 = exposure[early_ids].mean(axis=0)
    e1 = exposure[late_ids].mean(axis=0)
    n0 = counts[early_ids].mean(axis=0)
    n1 = counts[late_ids].mean(axis=0)
    if np.any(e0 <= 0) or np.any(e1 <= 0):
        raise ValueError('Zero exposure: conditional rate decomposition unidentifiable')
    r0, r1 = n0/e0, n1/e1
    env = (e1-e0)*(r0+r1)/2
    rate = (r1-r0)*(e0+e1)/2
    if not np.allclose(env+rate,n1-n0,atol=1e-10):
        raise RuntimeError('Decomposition identity failed')
    return e0,e1,r0,r1,env,rate,n0,n1


def analyze(frame, strata, repetitions=1000, block=1):
    keys = sorted(frame[strata].drop_duplicates().itertuples(index=False,name=None))
    exp = np.zeros((44,len(keys)))
    count = np.zeros_like(exp)
    for j,key in enumerate(keys):
        mask = np.ones(len(frame),dtype=bool)
        for col,val in zip(strata,key):
            mask &= frame[col].to_numpy() == val
        annual = frame.loc[mask].groupby('season_year')['events'].agg(['size','sum']).reindex(YEARS,fill_value=0)
        exp[:,j],count[:,j] = annual['size'],annual['sum']
    early,late = np.arange(22),np.arange(22,44)
    values = decompose(exp,count,early,late)
    rows=[]
    for j,key in enumerate(keys):
        row = dict(zip(strata,[int(v) for v in key]))
        row.update(dict(zip(['early_boxes_per_winter','late_boxes_per_winter','early_events_per_box',
                             'late_events_per_box','exposure_contribution_events_per_winter',
                             'rate_contribution_events_per_winter','early_events_per_winter',
                             'late_events_per_winter'],[float(x[j]) for x in values])))
        rows.append(row)
    rng = np.random.default_rng(SEED)
    samples=[]
    def draw(ids):
        starts=rng.integers(0,len(ids),size=int(np.ceil(len(ids)/block)))
        return ids[((starts[:,None]+np.arange(block))%len(ids)).ravel()[:len(ids)]]
    for _ in range(repetitions):
        try:
            val=decompose(exp,count,draw(early),draw(late))
        except ValueError:
            continue
        samples.append([val[4].sum(),val[5].sum(),(val[7]-val[6]).sum()])
    if len(samples)<.95*repetitions:
        raise RuntimeError('Too many nonidentifiable bootstrap samples')
    ci=np.quantile(samples,[.025,.975],axis=0)
    summary={name:{'estimate':float(estimate),'ci95_low':float(ci[0,i]),'ci95_high':float(ci[1,i])}
             for i,(name,estimate) in enumerate(zip(['exposure','conditional_rate','total'],
                             [values[4].sum(),values[5].sum(),(values[7]-values[6]).sum()]))}
    summary['valid_bootstrap_repetitions']=len(samples)
    summary['block_winters']=block
    return pd.DataFrame(rows),summary


def main():
    OUT.mkdir(exist_ok=True)
    times = target_times()
    winter = times.year + (times.month==12).astype(int)
    early = winter <= 2001
    a,p,lat,lon,sources=extract(times)
    outside=~((lat>=27)&(lat<=33)&(lon>=102)&(lon<=109))
    mask=outside & ((p[early]>=72500).mean(axis=0)>=.99)
    if mask.sum()<20:
        raise RuntimeError('Too few terrain-safe peripheral nodes')
    w=np.cos(np.deg2rad(lat[mask]))
    x=a[:,:,mask].astype(float)
    x[:,0,:]/=9.80665
    x[:,0,:]-=np.average(x[:,0,:],axis=1,weights=w)[:,None]
    keys=times.month*100+times.hour
    clim=np.empty_like(x)
    for key in np.unique(keys):
        clim[keys==key]=x[(keys==key)&early].mean(axis=0)
    anomaly=x-clim
    scales=np.sqrt((anomaly[early]**2).mean(axis=(0,2)))
    features=(anomaly/scales[None,:,None]*np.sqrt(w)[None,None,:]).reshape(len(times),-1)
    if not np.isfinite(features).all():
        raise RuntimeError('Nonfinite environmental features')
    pca=PCA(n_components=8,svd_solver='full')
    pca.fit(features[early])
    scores=pca.transform(features)
    km=KMeans(n_clusters=4,n_init=30,random_state=SEED).fit(scores[early])
    labels=km.predict(scores)
    meta=pd.read_csv(OUT/'events_terrain_qc_intensity.csv.gz')
    genesis=pd.to_datetime(meta.genesis_time,utc=True).dt.tz_localize(None)
    sy=genesis.dt.year+genesis.dt.month.eq(12).astype(int)
    selected=meta.track_cluster.eq(1)&genesis.dt.month.isin([12,1,2])&sy.between(1980,2023)
    counts=genesis[selected].dt.floor('6h').value_counts().reindex(times,fill_value=0)
    if counts.sum()!=selected.sum() or selected.sum()!=732:
        raise RuntimeError('Event target/count mismatch')
    frame=pd.DataFrame({'time':times,'season_year':winter,'month':times.month,'utc_hour':times.hour,
                        'regime':labels,'events':counts.to_numpy(),
                        'peripheral_valid_fraction':(p[:,mask]>=72500).mean(axis=1)})
    frame.to_csv(OUT/(PREFIX+'times.csv.gz'),index=False)
    annual=frame.groupby(['season_year','regime']).agg(exposure_boxes=('events','size'),events=('events','sum'))
    annual.to_csv(OUT/(PREFIX+'annual.csv'))
    table,summary=analyze(frame,['regime'])
    table.to_csv(OUT/(PREFIX+'decomposition.csv'),index=False)
    stratified,standard=analyze(frame,['regime','month','utc_hour'])
    stratified.to_csv(OUT/(PREFIX+'month_hour_decomposition.csv'),index=False)
    ev=pd.DataFrame({'season_year':sy[selected].to_numpy(),
                     'beijing_6h_bin':((genesis[selected].dt.hour+8)%24//6).to_numpy()})
    diurnal=ev.groupby(['season_year','beijing_6h_bin']).size().unstack(fill_value=0).reindex(YEARS,fill_value=0)
    diurnal.to_csv(OUT/(PREFIX+'diurnal_counts.csv'))
    np.savez_compressed(OUT/(PREFIX+'classification.npz'),lat=lat[mask],lon=lon[mask],
                        scales=scales,pca_components=pca.components_,pca_mean=pca.mean_,
                        cluster_centers=km.cluster_centers_,scores=scores,early=early,
                        raw_cluster_composites=np.stack([
                            x[early & (labels==k)].mean(axis=0) for k in range(4)]))
    result={'scope':'Exploratory catalogue-conditioned descriptive decomposition; not causal attribution',
            'early_winters':[1980,2001],'late_winters':[2002,2023],'n_events':int(selected.sum()),
            'n_boxes':len(times),'n_peripheral_nodes':int(mask.sum()),'antecedent_hours':24,
            'pca_explained_variance_fraction':float(pca.explained_variance_ratio_.sum()),
            'late_min_valid_fraction':float(frame.loc[~early,'peripheral_valid_fraction'].min()),
            'decomposition':summary,'month_hour_stratified':standard,
            'bootstrap':'1000 independent winter resamples within period; fixed classification, no training uncertainty',
            'seed':SEED,'input_files':sources,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/(PREFIX+'summary.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    text=['# 冬季局地生成机制：首轮天气型分解', '',
          '方案：`docs/winter_local_genesis_mechanism_plan_zh.md`。这是描述性、目录条件性探索结果。', '',
          f'事件：{selected.sum()}；六小时背景箱：{len(times)}；外围节点：{mask.sum()}；8个PC解释方差：{result["pca_explained_variance_fraction"]:.1%}。', '',
          '| 分解 | 变化/贡献（次/冬） | 固定分类95%区间 |', '|---|---:|---:|']
    for name,title in [('total','后期减前期'),('exposure','环境出现时长'),('conditional_rate','类型内生成率')]:
        r=summary[name]
        text.append(f'| {title} | {r["estimate"]:+.3f} | [{r["ci95_low"]:+.3f}, {r["ci95_high"]:+.3f}] |')
    text+=['', '月×UTC时次分层结果：',json.dumps(standard,ensure_ascii=False,indent=2), '',
           '不确定性尚未包含训练误差或跨年相关；晚期独立地形安全率见JSON。不能将生成率变化称为增暖贡献。',
           '下一步：核验天气类合成与地形有效率；开展提前量、类别数和块重采样敏感性；再选择动力预算候选。']
    (ROOT/'docs/winter_local_genesis_mechanism_results_zh.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['n_events','n_peripheral_nodes','late_min_valid_fraction','decomposition','month_hour_stratified']},indent=2),flush=True)


if __name__=='__main__':
    main()

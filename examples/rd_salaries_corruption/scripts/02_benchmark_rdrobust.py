"""Frequentist benchmark: reproduce the authors' Table 2 rdrobust specification exactly.
Requires the full analysis_full2.parquet (built by 01_prep.py from data-main.dta)."""
import numpy as np, pandas as pd, json
from rdrobust import rdrobust

df = pd.read_parquet('analysis_full2.parquet')
print('loaded', df.shape)

covars1 = [f'ccode{i}' for i in range(2,12)] + [f'year_{i}' for i in range(2,16)]
covars2 = covars1 + [f'threshold_{i}' for i in range(2,34)]

def run(mask, covs, label):
    d = df[mask].dropna(subset=['cri2','margin']).copy()
    cv = d[covs].astype(float)
    cv = cv.loc[:, cv.std(axis=0) > 0]
    out = rdrobust(y=d['cri2'].values, x=d['margin'].values, c=0,
                   covs=cv.values, cluster=d['citycode'].values, all=True)
    co = np.asarray(out.coef).flatten(); se = np.asarray(out.se).flatten(); pv = np.asarray(out.pv).flatten()
    h = float(np.asarray(out.bws).flatten()[0]); Nh = int(np.asarray(out.N_h).sum())
    res = dict(label=label, n=int(len(d)), N_in_bw=Nh, bandwidth=round(h,4),
               tau_conventional=round(float(co[0]),4),
               tau_bias_corrected=round(float(co[1]),4),
               se_robust=round(float(se[2]),4), p_robust=round(float(pv[2]),5))
    print(label, '->', res)
    return res

results=[]
results.append(run((df.uniques==1)&(df.margin<1), covars1, 'T2c1_unique_thresholds'))
results.append(run((df.non_uniques==1)&(df.margin<1), covars2, 'T2c2_compound_thresholds'))
results.append(run((df.margin<0.5), covars2, 'T2c3_margin_lt_0.5'))
json.dump(results, open('benchmark_results.json','w'), indent=2)
print('\nSAVED benchmark_results.json')

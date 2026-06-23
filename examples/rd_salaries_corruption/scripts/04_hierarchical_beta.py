"""
Stage 3 - 'hardcore Bayesian' RD: hierarchical, triangular-kernel-weighted,
Beta-likelihood local-linear RD. Improves on the paper's linear Gaussian model by
(i) respecting the bounded [0,1] support of cri2 via a Beta likelihood,
(ii) triangular kernel weights (Calonico-Cattaneo-Titiunik analogue),
(iii) city random intercepts (cluster-robust analogue) + country fixed effects,
(iv) separate slopes either side of the cutoff.
Reports the discontinuity tau on the natural cri2 scale (comparable to -0.078).
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import arviz as az, pymc as pm, pytensor.tensor as pt

H = 0.1115
d = pd.read_parquet("rd_unique.parquet").dropna(subset=["cri2","margin"]).copy()
d = d[d["margin"].abs() <= H].copy()
d["treated"] = (d["margin"] >= 0).astype(int)
N = len(d)
city_codes, city_idx = np.unique(d["citycode"].values, return_inverse=True)
ctry_codes, ctry_idx = np.unique(d["country"].astype(str).values, return_inverse=True)
nC = len(ctry_codes)
w = (1.0 - np.abs(d["margin"].values)/H)          # triangular kernel
y = d["cri2"].values.astype(float)
y2 = (y*(N-1)+0.5)/N                               # Smithson-Verkuilen squeeze into (0,1)
x = d["margin"].values.astype(float)
tr = d["treated"].values.astype(float)
print(f"hier sample N={N} cities={len(city_codes)} countries={nC} | {dict(zip(ctry_codes, np.bincount(ctry_idx)))}")

coords = {"city": city_codes, "country": ctry_codes}
with pm.Model(coords=coords) as m:
    a   = pm.Normal("a", -1.4, 1.0)               # logit baseline (~0.20)
    tau = pm.Normal("tau", 0.0, 1.0)              # discontinuity (logit scale)
    bL  = pm.Normal("bL", 0.0, 2.0)
    bR  = pm.Normal("bR", 0.0, 2.0)
    cfe = pm.ZeroSumNormal("cfe", sigma=1.0, dims="country") if nC > 1 else pt.zeros(1)
    sig_city = pm.HalfNormal("sig_city", 1.0)
    z_city = pm.Normal("z_city", 0.0, 1.0, dims="city")
    city_re = pm.Deterministic("city_re", z_city*sig_city, dims="city")
    phi = pm.Gamma("phi", alpha=3.0, beta=0.5)    # Beta precision, mean 6
    eta = (a + tau*tr + bL*x*(1-tr) + bR*x*tr
           + (cfe[ctry_idx] if nC>1 else 0.0) + city_re[city_idx])
    mu = pm.math.invlogit(eta)
    pm.Potential("lik", (pt.as_tensor_variable(w) *
                         pm.logp(pm.Beta.dist(mu*phi, (1-mu)*phi), y2)).sum())
    idata = pm.sample(draws=900, tune=900, chains=4, cores=4, target_accept=0.9,
                      random_seed=42, progressbar=False)

div = int(idata.sample_stats["diverging"].sum())
rhat = float(az.rhat(idata, var_names=["a","tau","bL","bR","phi","sig_city"]).to_array().max())
print("divergences", div, "max rhat", round(rhat,4))

post = idata.posterior
a_s   = post["a"].values.flatten()
tau_s = post["tau"].values.flatten()
inv = lambda z: 1/(1+np.exp(-z))
# country weights = composition near cutoff
cw = np.bincount(ctry_idx, minlength=nC)/N
if nC > 1:
    cfe_s = post["cfe"].stack(s=("chain","draw")).transpose("s","country").values  # (S,nC)
    p1 = inv(a_s[:,None] + tau_s[:,None] + cfe_s)         # treated, by country
    p0 = inv(a_s[:,None] + cfe_s)
    tau_cri2 = (p1 - p0) @ cw                             # weighted over country mix
else:
    tau_cri2 = inv(a_s+tau_s) - inv(a_s)

def summ(s):
    return dict(mean=round(float(s.mean()),4), sd=round(float(s.std()),4),
                hdi3=round(float(np.percentile(s,3)),4), hdi97=round(float(np.percentile(s,97)),4),
                p_negative=round(float((s<0).mean()),4))
out = {"model":"hierarchical_beta_kernel_local_linear","bandwidth":H,"N":N,
       "n_cities":int(len(city_codes)),"n_countries":int(nC),
       "divergences":div,"max_rhat":round(rhat,4),
       "tau_logit": summ(tau_s), "tau_cri2_scale": summ(tau_cri2),
       "baseline_cri2_control": round(float(inv(a_s).mean()),4)}
json.dump(out, open("out/results_hier.json","w"), indent=2)
np.save("out/post_hier_cri2.npy", tau_cri2)
print(json.dumps(out, indent=2))

fig, ax = plt.subplots(1,2, figsize=(11,4))
az.plot_posterior(tau_s, ref_val=0, hdi_prob=0.94, ax=ax[0]); ax[0].set_title("Discontinuity tau (logit scale)")
az.plot_posterior(tau_cri2, ref_val=0, hdi_prob=0.94, ax=ax[1]); ax[1].set_title("Discontinuity on cri2 scale")
for a_ in ax: a_.axvline(0, color="k", ls=":")
ax[1].axvline(-0.078, color="red", ls="--", label="paper -0.078"); ax[1].legend()
fig.suptitle("Hierarchical Beta kernel-weighted RD: effect of salary raise on corruption risk")
fig.savefig("out/fig_hier_posterior.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print("SAVED out/results_hier.json + fig_hier_posterior.png")

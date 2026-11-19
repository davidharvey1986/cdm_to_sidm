"""Use all four runs of a cluster (CDM, SIDM 0.1, 0.3, 1) together instead of separate CDM-SIDM pairs.

1. scale-band correlation matrix of the 4 runs (is the run-to-run noise independent between runs?)
2. per cluster, per pixel: delta_sigma = a + b * sigma^p  (3 sigmas, p from the BAHAMAS template)
   a = offset shared by all three deltas (the CDM run's own noise enters every delta with the same sign),
   b = the cluster's SIDM response at sigma = 1.
3. cluster-to-cluster scatter of the response: var(b_hat) = var_true + noise, noise from the fit residuals.
Output runs/sigma_sequence/summary.json
usage: python3 sigma_sequence.py
"""
import os, json
import numpy as np
from scipy.ndimage import gaussian_filter as G
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/runs/sigma_sequence"; os.makedirs(OUT, exist_ok=True)
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig, sh, spl, M, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "log10_M200", "redshift"))
SIG = (0.1, 0.3, 1.0)
ok = (np.abs(sh).max(1) <= 2) & (sig > 0)
rows = {}
for i in np.where(ok)[0]:
    rows.setdefault(xi[i], {})[float(sig[i])] = i
cl = np.array([c for c, d in rows.items() if all(s in d for s in SIG)])
print(f"{len(cl)} CDM maps with all three SIDM runs (|shift| <= 2)")
X = x[cl, 2:98, 2:98].astype(np.float64)
Y = np.stack([y[[rows[c][s] for c in cl], 2:98, 2:98] for s in SIG], 1).astype(np.float64)  # (n,3,96,96)
runs = np.concatenate([X[:, None], Y], 1)  # CDM, 0.1, 0.3, 1
n = 96; yy, xx = np.indices((n, n)); r = np.hypot(yy - 47.5, xx - 47.5); RING = (r > 8) & (r < 45); CORE = r < 2
res = {"n_clusters": int(len(cl))}

# 1. band correlations
names = ["CDM", "0.1", "0.3", "1"]
for lp, hp, lab in ((0, 3, "<3 px"), (1.5, 8, "1.5-8 px"), (4, 20, "4-20 px")):
    B = G(runs, (0, 0, lp, lp)) - G(runs, (0, 0, hp, hp)) if lp > 0 else runs - G(runs, (0, 0, hp, hp))
    B = B[:, :, RING]; B = B - B.mean(-1, keepdims=True); B /= np.linalg.norm(B, axis=-1, keepdims=True)
    C = np.median(np.einsum("nip,njp->nij", B, B), 0)
    res[f"corr_{lab}"] = C.round(3).tolist()
    print(f"band {lab}: median correlation between runs (rows/cols CDM, 0.1, 0.3, 1)\n{np.array2string(C, precision=3)}")

# 2. per-pixel fit delta_s = a + b s^p
p = json.load(open(f"{HERE}/runs/two_scale_bahamas_nocnn/template.json"))["template"]["p"]
w = np.array(SIG) ** p; Dl = Y - X[:, None]                      # (n,3,96,96)
A_ = np.stack([np.ones(3), w], 1); Pinv = np.linalg.pinv(A_)    # (2,3)
coef = np.einsum("ks,nsij->nkij", Pinv, Dl); a, b = coef[:, 0], coef[:, 1]
fit = np.einsum("sk,nkij->nsij", A_, coef); resid = Dl - fit
dof = 3 - 2
s2 = (resid ** 2).sum(1) / dof                                   # per-pixel noise variance of one delta
covb = np.linalg.inv(A_.T @ A_)                                   # noise variance factors for (a, b)
core = lambda m: (m * CORE).sum((-2, -1)) / CORE.sum()
b_core, a_core = core(b), core(a)
# noise in the core-averaged b: from residuals of the core-averaged deltas (pixels in the core are correlated)
rc = core(resid); s2c = (rc ** 2).sum(1) / dof
noise_b = float(np.mean(s2c) * covb[1, 1]); noise_a = float(np.mean(s2c) * covb[0, 0])
var_b = float(b_core.var()); var_a = float(a_core.var())
res.update(p=p, b_core_median=float(np.median(b_core)), b_core_std=float(np.sqrt(var_b)),
           b_core_noise_std=float(np.sqrt(noise_b)), b_core_true_std=float(np.sqrt(max(var_b - noise_b, 0))),
           a_core_median=float(np.median(a_core)), a_core_std=float(np.sqrt(var_a)), a_core_noise_std=float(np.sqrt(noise_a)))
# single-pair core scatter for comparison
for k, s in enumerate(SIG):
    res[f"single_pair_core_std_{s}"] = float(core(Dl[:, k]).std())
print(f"\nfit delta_sigma = a + b sigma^{p:.3f}, core r<2 px:")
print(f"  b (response at sigma=1): median {res['b_core_median']:+.3f}, scatter {res['b_core_std']:.3f} = noise {res['b_core_noise_std']:.3f} (+) intrinsic {res['b_core_true_std']:.3f}")
print(f"  a (shared offset):       median {res['a_core_median']:+.3f}, scatter {res['a_core_std']:.3f}")
print("  single-pair core delta scatter: " + "  ".join(f"sigma {s}: {res[f'single_pair_core_std_{s}']:.3f}" for s in SIG))
# fraction of delta variance explained by the shared offset a (whole field)
for k, s in enumerate(SIG):
    v0 = Dl[:, k][:, RING].var(); v1 = (Dl[:, k] - a)[:, RING].var()
    res[f"field_var_removed_by_a_{s}"] = float(1 - v1 / v0)
print("  field (8<r<45) variance of each delta removed by subtracting the shared offset a: " +
      "  ".join(f"{s}: {res[f'field_var_removed_by_a_{s}']:.0%}" for s in SIG))
# 3. what predicts b?
Mc, zc = M[cl], zz[cl]; cen = core(X); slope = core(X) - (X * ((r > 4) & (r < 8))).sum((1, 2)) / ((r > 4) & (r < 8)).sum()
for nm, v in (("log M200", Mc), ("z", zc), ("CDM central log Sigma", cen), ("CDM inner slope (core - 4-8px)", slope)):
    rho = spearmanr(v, b_core)[0]; res[f"spearman_b_vs_{nm}"] = float(rho)
    print(f"  spearman(b_core, {nm}) = {rho:+.2f}")
# split by halves: is b reproducible? correlate b from (0.1,1) vs from (0.3,1)? (shares run 1) -> use a,b noise instead
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
np.savez_compressed(f"{OUT}/fit.npz", cdm_row=cl, a=a.astype(np.float32), b=b.astype(np.float32), resid_var=s2.astype(np.float32))
print("wrote", OUT)

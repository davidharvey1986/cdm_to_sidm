"""Test 1: does the per-pixel velocity dispersion predict the SIDM change beyond density, radius, mass and redshift?

Target (SIDM-specific, CDM noise cancels): d = log Sigma_SIDM(sigma_hi) - log Sigma_SIDM(0.1) of the same cluster
(both stamps already re-centred on CDM; sigma_hi = 1 and 0.3), smoothed 1.5 px like the velocity moments.
Features from the CDM run only (so no selection on the SIDM runs' noise):
  base   : log Sigma_tot (CDM stamp), log Sigma_DM (±R200), r / R200, log M200, z
  +vdisp : base + log sigma_los (±R200)
  +proxy : base + log (Sigma_DM sigma_los)  [same information as +vdisp, for completeness]
Gradient-boosted trees (sklearn HistGradientBoostingRegressor), trained on pixels of 'train'-split clusters and scored
on 'test'-split clusters (pairs.npz split), central 96x96 px, |shift| <= 2 for both SIDM maps.
Also: mean d against the sigma_los residual (log sigma_los minus its mean at fixed r/R200, M, z) in density bins.
usage: python3 vdisp_test.py -> unet/figures/vdisp_test.png, runs/vdisp_test.json
"""
import os, json, numpy as np, matplotlib
from scipy.ndimage import gaussian_filter as G
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); UNET = os.path.join(HERE, "..", "..", "unet")
D = np.load(f"{UNET}/data/pairs.npz", allow_pickle=True); X = np.load(f"{UNET}/data/sidm_proxy_bahamas.npz")
x, y, xi, sig, sh, spl, M, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "log10_M200", "redshift"))
C = 1.989e33 / 3.0857e24 ** 2; rng = np.random.default_rng(0)
rho_c = lambda z: 2.775e11 * 0.7 ** 2 * (0.3 * (1 + z) ** 3 + 0.7)  # Msun / Mpc^3
r200 = (3 * 10.0 ** M / (4 * np.pi * 200 * rho_c(zz))) ** (1 / 3)  # pMpc
yy, xx = np.indices((96, 96)); RPX = np.hypot(yy - 47.5, xx - 47.5) * 0.02  # pMpc
ok = np.abs(sh).max(1) <= 2; rows = {}
for i in np.where(ok & (sig > 0))[0]: rows.setdefault(xi[i], {})[float(sig[i])] = i
good = np.isfinite(X["base_slab"]).all((1, 2))
res, figdat = {}, {}
for shi in (1.0, 0.3):
    cl = np.array([c for c, d_ in rows.items() if 0.1 in d_ and shi in d_ and good[c]])
    NPX = 400  # pixels sampled per cluster (all clusters, all radii)
    F, T, S = [], [], []
    for c in cl:
        d = G(y[rows[c][shi]].astype(float) - y[rows[c][0.1]].astype(float), 1.5)[2:98, 2:98]
        sv = X["sigma_los_slab"][c][2:98, 2:98]; Sd = X["Sigma_dm_slab"][c][2:98, 2:98] * C
        p = rng.choice(96 * 96, NPX, replace=False); f_ = lambda a: a.ravel()[p]
        F.append(np.c_[f_(x[c][2:98, 2:98]), np.log10(f_(Sd)), f_(RPX) / r200[c], np.full(NPX, M[c]), np.full(NPX, zz[c]), np.log10(f_(sv))])
        T.append(f_(d)); S.append(np.full(NPX, spl[c]))
    F, T, S = np.concatenate(F), np.concatenate(T), np.concatenate(S)
    keep = np.isfinite(F).all(1) & np.isfinite(T); F, T, S = F[keep], T[keep], S[keep]
    tr, te = S == "train", S == "test"
    out = {}
    for name, cols in (("base", [0, 1, 2, 3, 4]), ("+vdisp", [0, 1, 2, 3, 4, 5])):
        m = HGB(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200, random_state=0).fit(F[tr][:, cols], T[tr])
        pr = m.predict(F[te][:, cols]); mse = np.mean((pr - T[te]) ** 2)
        out[name] = dict(r2=float(1 - mse / T[te].var()), rmse=float(np.sqrt(mse)))
        # inner region only (r < 0.3 R200), where SIDM acts
        inn = F[te][:, 2] < 0.3; out[name]["r2_inner"] = float(1 - np.mean((pr[inn] - T[te][inn]) ** 2) / T[te][inn].var())
    # bootstrap over test clusters for the R2 gain
    res[f"{shi}-0.1"] = dict(n_clusters=int(len(cl)), n_train_px=int(tr.sum()), n_test_px=int(te.sum()), **out,
                             gain_r2=out["+vdisp"]["r2"] - out["base"]["r2"], gain_r2_inner=out["+vdisp"]["r2_inner"] - out["base"]["r2_inner"])
    # sigma residual at fixed r/R200, M, z (from a fit of log sigma_los on those)
    mr = HGB(max_iter=200, random_state=0).fit(F[tr][:, [2, 3, 4]], F[tr][:, 5]); resid = F[:, 5] - mr.predict(F[:, [2, 3, 4]])
    figdat[shi] = (F, T, resid)
    print(f"SIDM{shi} - SIDM0.1: {len(cl)} clusters, {tr.sum()} train / {te.sum()} test px; "
          f"R2 base {out['base']['r2']:.4f} -> +vdisp {out['+vdisp']['r2']:.4f}; inner (r<0.3 R200) {out['base']['r2_inner']:.4f} -> {out['+vdisp']['r2_inner']:.4f}", flush=True)
json.dump(res, open(f"{UNET}/runs/vdisp_test.json", "w"), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(20, 5.3), constrained_layout=True)
F, T, resid = figdat[1.0]
for (lo, hi), cl_ in zip(((-3.0, -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, 0.5)), ("#3a6fb0", "#2a9d8f", "#e9a03b", "#c0392b")):
    s = (F[:, 1] >= lo) & (F[:, 1] < hi); b = np.linspace(-0.25, 0.25, 21); mb = 0.5 * (b[1:] + b[:-1]); idx = np.digitize(resid[s], b) - 1
    mu = np.array([T[s][idx == q].mean() if (idx == q).sum() > 300 else np.nan for q in range(len(mb))])
    er = np.array([T[s][idx == q].std() / np.sqrt((idx == q).sum()) if (idx == q).sum() > 300 else np.nan for q in range(len(mb))])
    ax[0].errorbar(mb, mu - np.nanmean(mu), er, fmt="o-", ms=3, color=cl_, capsize=0, label=f"log Σ_DM {lo} to {hi}")
ax[0].axhline(0, color="0.5", lw=0.7); ax[0].set_xlabel("σ_los residual: log σ_los − mean at fixed r/R200, M, z")
ax[0].set_ylabel("mean d (offset per density bin)"); ax[0].legend(fontsize=8); ax[0].set_title("SIDM1 − SIDM0.1: change vs σ_los residual, at fixed density")
for (lo, hi), cl_ in zip(((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 2)), ("#c0392b", "#e9a03b", "#2a9d8f", "#3a6fb0")):
    s = (F[:, 2] >= lo) & (F[:, 2] < hi); b = np.linspace(np.percentile(F[s, 5], 1), np.percentile(F[s, 5], 99), 16); mb = 0.5 * (b[1:] + b[:-1]); idx = np.digitize(F[s, 5], b) - 1
    mu = [T[s][idx == q].mean() if (idx == q).sum() > 300 else np.nan for q in range(len(mb))]
    ax[1].plot(10 ** mb, mu, "o-", ms=3, color=cl_, label=f"r/R200 {lo}–{hi}")
ax[1].axhline(0, color="0.5", lw=0.7); ax[1].set_xlabel("σ_los [km/s]"); ax[1].set_ylabel("mean d"); ax[1].legend(fontsize=8)
ax[1].set_title("SIDM1 − SIDM0.1: change vs σ_los, by radius (raw)")
lab = list(res); w = 0.35; xs = np.arange(len(lab))
for j, (k_, nm) in enumerate((("r2", "all pixels"), ("r2_inner", "r < 0.3 R200"))):
    ax[2].bar(xs + (j - 0.5) * w - 0.08, [res[l]["base"][k_] for l in lab], 0.16, color="0.6" if j == 0 else "0.3", label=f"base, {nm}")
    ax[2].bar(xs + (j - 0.5) * w + 0.08, [res[l]["+vdisp"][k_] for l in lab], 0.16, color="#2a9d8f" if j == 0 else "#1b6b61", label=f"+σ_los, {nm}")
ax[2].set_xticks(xs, [f"SIDM{l.split('-')[0]} − SIDM0.1" for l in lab]); ax[2].set_ylabel("held-out R² of d"); ax[2].legend(fontsize=8)
ax[2].set_title("held-out clusters: does adding σ_los help?")
fig.suptitle("Test 1: SIDM-specific change (two SIDM runs, CDM noise cancels) vs CDM velocity dispersion at fixed density / radius / mass / z")
fig.savefig(f"{UNET}/figures/vdisp_test.png", dpi=90); print(json.dumps({k: (round(v['gain_r2'], 4), round(v['gain_r2_inner'], 4)) for k, v in res.items()}))

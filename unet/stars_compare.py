"""Evaluate the stars flow on the test split: runs/stars_true (oracle SIDM DM) and runs/stars_hyb (DM from the hybrid flow).
Stars maps are log10(Sigma_star + 1e11), 96x96, 20 kpc px; Delta = log SIDM - log CDM.
Figure 1 (figures/stars_compare_stats.png), rows sigma:
  stacked profile ratio <Sigma_SIDM>/<Sigma_CDM>(R); pixel rms of Delta(R) (one draw); PIT histogram (R < 46 px);
  per-pixel corr(Delta_stars, Delta_DM)(R) with the DM change each was conditioned on; per-pair stellar-mass change < 50 kpc.
Figure 2 (figures/stars_compare_maps.png): three SIDM1 test clusters: CDM, true SIDM, two oracle draws, one chain draw, and Delta.
usage: python stars_compare.py
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dm_common as C, two_scale as T2
S, _ = C.load(); te = S["test"]; D = np.load(f"{C.HERE}/data/pairs_stars.npz", allow_pickle=True)
M = {}
for n, dmk in (("oracle", "true"), ("chain", "hyb")):
    f = f"{C.HERE}/runs/stars_{dmk}/draws.npz"
    if os.path.isfile(f): d = np.load(f); M[n] = (d["draws"].astype(np.float32), d["k"]); print(n, d["draws"].shape, "epoch", d["epoch"])
NK = min(len(v[1]) for v in M.values()); k = te["k"][:NK]
xs = D["x"][D["xi"][k]][:, 2:98, 2:98]; ys = D["y"][k][:, 2:98, 2:98]; cdm, sdm, sig = te["c"][:NK], te["s"][:NK], te["sigma"][:NK]
dmh = np.load(f"{C.HERE}/runs/dm_flow_hyb/draws.npz")["draws"][:, :NK].astype(np.float32)
LAB = {"oracle": "flow, true SIDM DM", "chain": "flow, DM-flow draw"}; COL = {"oracle": "#2a9d8f", "chain": "#d1495b"}
RB = T2.RB; cnt = np.bincount(RB.ravel()); Rk = np.arange(48) * 20 + 10; A50 = T2.R < 2.5; FIELD = T2.FIELD
ring = lambda m: np.bincount(RB.ravel(), m.ravel(), 48)
lin = lambda l: np.maximum(10 ** l.astype(np.float64) - 1e11, 0)
def corr_prof(d, dd):
    return np.array([np.corrcoef(d[:, RB == r].ravel(), dd[:, RB == r].ravel())[0, 1] for r in range(48)])
fig, ax = plt.subplots(3, 5, figsize=(28, 15), constrained_layout=True); summ = {}
for i, s in enumerate(C.SIG):
    m = sig == s; x, t = xs[m], ys[m]; dT = t - x; A = ax[i]; summ[str(s)] = {}
    lx = lin(x).sum(0); A[0].plot(Rk, ring(lin(t).sum(0)) / ring(lx), "k", lw=2.4, label="true")
    A[1].plot(Rk, np.sqrt(ring((dT ** 2).mean(0)) / cnt), "k", lw=2.4, label="true")
    A[3].plot(Rk, corr_prof(dT, sdm[m] - cdm[m]), "k", lw=2.4, label="true")
    m50 = lambda q: np.log10(lin(q)[:, A50].sum(1) / lin(x)[:, A50].sum(1)); bins = np.linspace(-0.5, 0.5, 51)
    A[4].hist(m50(t), bins, histtype="step", color="k", lw=2.4, label=f"true (sd {m50(t).std():.3f})")
    summ[str(s)]["sim"] = dict(core_ratio=float((ring(lin(t).sum(0)) / ring(lx))[0]), rms_core=float(np.sqrt((dT[:, A50] ** 2).mean())), m50_sd=float(m50(t).std()),
                                peaks=float(C.peaks(t)[0].mean()))
    for n, (X, _) in M.items():
        Xm = X[:, :NK][:, m]; dX = Xm - x[None]; cl = COL[n]
        A[0].plot(Rk, ring(lin(Xm).sum((0, 1)) / len(Xm)) / ring(lx), color=cl, lw=1.8, ls="--", label=LAB[n])
        A[1].plot(Rk, np.sqrt(ring((dX[0] ** 2).mean(0)) / cnt), color=cl, lw=1.8, ls="--", label=LAB[n])
        rank = (Xm < t[None]).sum(0); h = np.bincount(rank[:, FIELD].ravel(), minlength=len(Xm) + 1) / rank[:, FIELD].size
        A[2].plot(np.arange(len(Xm) + 1), h * (len(Xm) + 1), color=cl, lw=2, marker="o", ms=3, label=LAB[n])
        ddm = (sdm[m] if n == "oracle" else dmh[0][m]) - cdm[m]; A[3].plot(Rk, corr_prof(dX[0], ddm), color=cl, lw=1.8, ls="--", label=LAB[n])
        A[4].hist(m50(Xm[0]), bins, histtype="step", color=cl, lw=1.8, ls="--", label=f"{LAB[n]} (sd {m50(Xm[0]).std():.3f})")
        cov = float(np.mean((rank[:, FIELD] >= 0.05 * (len(Xm) + 1) - 0.5) & (rank[:, FIELD] <= 0.95 * (len(Xm) + 1) - 0.5)))
        summ[str(s)][n] = dict(core_ratio=float((ring(lin(Xm).sum((0, 1)) / len(Xm)) / ring(lx))[0]), rms_core=float(np.sqrt((dX[0][:, A50] ** 2).mean())),
                               m50_sd=float(m50(Xm[0]).std()), peaks=float(C.peaks(Xm[0])[0].mean()), central90=cov,
                               pit_chi2=float(((h * (len(Xm) + 1) - 1) ** 2).sum()))
    A[2].axhline(1, color="k", lw=0.8, ls=":"); A[0].axhline(1, color="0.6", lw=0.6, ls=":"); A[3].axhline(0, color="0.6", lw=0.6)
    A[0].set_ylabel(f"SIDM{s:g} ({m.sum()} test maps)\n<Σ_SIDM> / <Σ_CDM>", fontsize=12); A[1].set_ylabel("rms Δ [dex]")
    A[2].set_ylabel("PIT density"); A[2].set_xlabel("rank of true pixel among draws"); A[3].set_ylabel("corr(Δ stars, Δ DM)")
    A[4].set_xlabel("log10 M*_SIDM / M*_CDM, R < 50 kpc")
    for j in (0, 1, 3): A[j].set_xscale("log"); A[j].set_xlabel("R [kpc]")
    for a_ in A: a_.legend(fontsize=8)
for j, t_ in enumerate(["stacked stellar profile ratio", "pixel rms of Δ (one draw)", "calibration (PIT)", "do stars follow the DM change?", "central stellar-mass change"]):
    ax[0, j].set_title(t_, fontsize=13)
fig.suptitle("Stars flow, BAHAMAS test split: true SIDM stars vs draws (oracle DM, and DM from the hybrid DM flow)", fontsize=15)
fig.savefig(f"{C.HERE}/figures/stars_compare_stats.png", dpi=65); json.dump(summ, open(f"{C.HERE}/runs/stars_compare_summary.json", "w"), indent=1)
for s, v in summ.items():
    for n, r in v.items(): print(f"SIDM{s} {n:5s} " + " ".join(f"{q}={w:.3f}" for q, w in r.items()))
# example maps
P1 = np.where(sig == 1.0)[0]; lm = te["z"][:0]
ex = [P1[q] for q in np.linspace(0, len(P1) - 1, 3).astype(int)]
cols = [("CDM", lambda j: xs[j]), ("true SIDM1", lambda j: ys[j])] + \
       [(f"{LAB[n]} {q + 1}", (lambda n, q: lambda j: M[n][0][q, j])(n, q)) for n in M for q in ((0, 1) if n == "oracle" else (0,))]
fig, ax = plt.subplots(2 * len(ex), len(cols), figsize=(3.8 * len(cols), 7.4 * len(ex)), constrained_layout=True)
for r, j in enumerate(ex):
    for c, (t_, f) in enumerate(cols):
        im = f(j); ax[2 * r, c].imshow(im, origin="lower", cmap="magma", vmin=11, vmax=15.5); ax[2 * r, c].set_title(t_, fontsize=11)
        ax[2 * r + 1, c].imshow(im - xs[j], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        for q in (0, 1): ax[2 * r + q, c].set_xticks([]); ax[2 * r + q, c].set_yticks([])
    ax[2 * r, 0].set_ylabel(f"test pair {j}, z {te['z'][j]:.3f}", fontsize=11); ax[2 * r + 1, 0].set_ylabel("− CDM (±1 dex)", fontsize=11)
fig.suptitle("Stars flow: SIDM1 test clusters, maps and change from CDM", fontsize=15)
fig.savefig(f"{C.HERE}/figures/stars_compare_maps.png", dpi=50); print("wrote figures/stars_compare_stats.png, stars_compare_maps.png")

"""Halo shape test: projected ellipticity profile eps(R) = 1 - q(R) of the dark-matter maps, BAHAMAS test split.

Shape within R: iterative mass-weighted second moments of Sigma (linear) inside an ellipse of semi-major axis R about the
stamp centre (pixels with x^2 + y^2/q^2 < R^2 in the principal frame), starting circular and updating q and the position
angle until converged (NIT iterations). q = sqrt(lambda_min / lambda_max) of the moment tensor.
Models: CDM (test-split CDM maps), true SIDM0.1/0.3/1, and draws of the DM flows (runs/dm_<name>/draws.npz; NDRAW draws each).
Figure figures/dm_shape.png: (1) mean eps(R) for the four simulated models; (2-4) per sigma: true SIDM vs flow draws, CDM
for reference; (5) per-cluster shape change eps_SIDM1 - eps_CDM at three radii, true vs flow_hyb.
usage: python dm_shape.py [flow flow_hyb]
"""
import sys, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dm_common as C
from halo_shape import RK, shapes
NDRAW = 4
names = sys.argv[1:] or ["flow", "flow_hyb"]
S, _ = C.load(); te = S["test"]; sig = te["sigma"]
_, first = np.unique(te["row"], return_index=True)  # one CDM map per cluster-projection
eC = shapes(te["c"]); eT = shapes(te["s"]); print("true maps done", flush=True)
eD = {}
for n in names:
    d = np.load(f"{C.HERE}/runs/dm_{n}/draws.npz"); assert (d["k"] == te["k"]).all()
    eD[n] = np.stack([shapes(d["draws"][q].astype(np.float32)) for q in range(NDRAW)]); print(n, "done", flush=True)
COL = {"CDM": "#222222", 0.1: "#3a6fb0", 0.3: "#2a9d8f", 1.0: "#d1495b"}; DC = dict(zip(names, ["#e9a03b", "#8e44ad", "#c0392b"]))
mse = lambda e: (e.mean(0), e.std(0) / np.sqrt(len(e)))
fig, ax = plt.subplots(1, 5, figsize=(29, 5.6), constrained_layout=True); summ = {}
mC, sC = mse(eC[first]); ax[0].plot(RK, mC, color=COL["CDM"], lw=2.4, label=f"CDM ({len(first)} maps)"); ax[0].fill_between(RK, mC - sC, mC + sC, color=COL["CDM"], alpha=0.2, lw=0)
summ["CDM"] = mC.tolist()
for i, s in enumerate(C.SIG):
    m = sig == s; mT, sT = mse(eT[m]); mc, sc = mse(eC[m])
    ax[0].plot(RK, mT, color=COL[s], lw=2.4, label=f"SIDM{s:g}"); ax[0].fill_between(RK, mT - sT, mT + sT, color=COL[s], alpha=0.2, lw=0)
    A = ax[i + 1]
    A.plot(RK, mc, color=COL["CDM"], lw=1.6, ls=":", label="CDM"); A.plot(RK, mT, color=COL[s], lw=2.6, label=f"true SIDM{s:g}")
    A.fill_between(RK, mT - sT, mT + sT, color=COL[s], alpha=0.2, lw=0); summ[f"SIDM{s:g}"] = {"true": mT.tolist(), "err": sT.tolist()}
    for n, E in eD.items():
        mD = E[:, m].mean((0, 1)); A.plot(RK, mD, color=DC[n], lw=1.8, ls="--", label=n); summ[f"SIDM{s:g}"][n] = mD.tolist()
    A.set_title(f"SIDM{s:g} ({m.sum()} test maps)", fontsize=13)
# per-cluster change at three radii, SIDM1
m = sig == 1.0; bins = np.linspace(-0.3, 0.3, 61)
for j, (R, ls) in enumerate(zip((60, 200, 600), ("-", "--", ":"))):
    jj = int(np.where(RK == R)[0][0]); dT = eT[m, jj] - eC[m, jj]
    ax[4].hist(dT, bins, histtype="step", color="k", lw=2, ls=ls, label=f"true, R={R} (mean {dT.mean():+.3f}, sd {dT.std():.3f})")
    if "flow_hyb" in eD:
        dF = eD["flow_hyb"][0, m, jj] - eC[m, jj]
        ax[4].hist(dF, bins, histtype="step", color=DC["flow_hyb"], lw=2, ls=ls, label=f"flow_hyb (mean {dF.mean():+.3f}, sd {dF.std():.3f})")
        print(f"SIDM1 R={R}: true d_eps mean {dT.mean():+.4f} sd {dT.std():.4f}; flow_hyb mean {dF.mean():+.4f} sd {dF.std():.4f}; "
              f"corr(true, flow_hyb) {np.corrcoef(dT, dF)[0, 1]:.2f}")
ax[4].set_xlabel("ε_SIDM1 − ε_CDM per cluster"); ax[4].set_title("per-cluster shape change, SIDM1", fontsize=13); ax[4].legend(fontsize=8)
for a in ax[:4]: a.set_xscale("log"); a.set_xlabel("R (semi-major) [kpc]"); a.set_ylabel("mean ε = 1 − b/a"); a.legend(fontsize=9)
ax[0].set_title("simulations: CDM vs SIDM (test split)", fontsize=13)
fig.suptitle("Projected dark-matter halo shape: mean ellipticity within R (iterative second moments), BAHAMAS test split", fontsize=15)
fig.savefig(f"{C.HERE}/figures/dm_shape.png", dpi=65); json.dump(dict(R_kpc=RK.tolist(), **summ), open(f"{C.HERE}/runs/dm_shape_summary.json", "w"), indent=1)
print("R [kpc]       ", " ".join(f"{r:6d}" for r in RK)); print("CDM           ", " ".join(f"{v:6.3f}" for v in mC))
for s in C.SIG:
    d = summ[f"SIDM{s:g}"]
    for n in ["true"] + names: print(f"SIDM{s:g} {n:9s}", " ".join(f"{v:6.3f}" for v in d[n]))
print("wrote figures/dm_shape.png")

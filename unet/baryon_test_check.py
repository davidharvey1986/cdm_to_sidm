"""Check the COSMA test gas + stellar particle maps (velmaps/bahamas/test_baryons, CDM z=0, fof 0-4):
does binned DM + gas + stars reproduce the original (IDL, SPH-projected) total map of the stamp?
Columns: original total, particle sum (DM+gas+stars), log ratio, particle gas, particle stars, radial fractions.
All maps smoothed 1.5 px. Output figures/baryon_test_check.png
"""
import os, json, pickle
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as G
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"; V = f"{IMG}/velmaps_v2_imaged/CDM+baryons/snap_032"
BT = f"{HERE}/../velmaps/bahamas/test_baryons"
O = json.load(open(f"{HERE}/../velmaps/bahamas/orientation.json"))
meta, stamps = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb"))
SM = 1.5
orient = lambda m, o: (lambda r: r[:, ::-1] if o["flip"] else r)(np.rot90(m, o["rot"]))
sel = [k for k in range(len(meta["redshift"])) if meta["redshift"][k] == 0 and 0 <= meta["fof_cdm"][k] <= 4 and meta["mask"][k, 0]]
sel = sorted(sel, key=lambda k: (meta["fof_cdm"][k], meta["projection"][k]))[:6]
Y, X = np.indices((100, 100)); R = np.hypot(X - 49.5, Y - 49.5).astype(int); rk = (np.arange(50) + 0.5) * 20
prof = lambda m: np.bincount(R.ravel(), m.ravel())[:50] / np.bincount(R.ravel())[:50]
fig, ax = plt.subplots(len(sel), 6, figsize=(24, 4 * len(sel)), constrained_layout=True); rows = []
for r, k in enumerate(sel):
    fof, P = int(meta["fof_cdm"][k]), int(meta["projection"][k]); o = O[f"projection_{P}"]
    dv = np.load(f"{V}/los{o['los']}.npz"); db = np.load(f"{BT}/los{o['los']}.npz")
    dm = orient(dv["moments"][int(np.where(dv["fof"] == fof)[0][0]), 0].astype(np.float64), o)
    ib = int(np.where(db["fof"] == fof)[0][0]); gas = orient(db["maps"][ib, 0].astype(np.float64), o); st = orient(db["maps"][ib, 2].astype(np.float64), o)
    tot = stamps[k, 0].astype(np.float64)
    tS, sumS, gS, sS, dS = (G(m, SM) for m in (tot, dm + gas + st, gas, st, dm))
    lr = np.log10(sumS / tS); inner = R < 45; rows.append((fof, P, np.median(lr[inner]), lr[inner].std(), (gS / tS)[R < 25].mean()))
    for c, (m, t) in enumerate([(tS, "original total (IDL)"), (sumS, "DM + gas + stars (particles)")]):
        im = ax[r, c].imshow(np.log10(m), origin="lower", cmap="magma", vmin=12.5, vmax=15.8)
    ax[r, 2].imshow(lr, origin="lower", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax[r, 2].set_title(("log10(sum / original)\n" if r == 0 else "") + f"median {rows[-1][2]:+.3f}, rms {rows[-1][3]:.3f} (R<0.9 Mpc)", fontsize=10)
    ax[r, 3].imshow(np.log10(np.maximum(gS, 1e11)), origin="lower", cmap="magma", vmin=12.5, vmax=15.0)
    ax[r, 4].imshow(np.log10(np.maximum(sS, 1e10)), origin="lower", cmap="magma", vmin=12.5, vmax=15.8)
    for c in range(5): ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    for c, t in enumerate(["original total (IDL)", "DM + gas + stars (particles)", None, "gas (particles)", "stars (particles)"]):
        if r == 0 and t: ax[r, c].set_title(t)
    ax[r, 0].set_ylabel(f"CDM z=0 fof {fof} proj {P}\nlogM {meta['log10_M200'][k]:.2f}", fontsize=11)
    pt = prof(tS)
    for m, t, cl in [(dS, "DM", "#3a6fb0"), (gS, "gas", "#2a9d8f"), (sS, "stars", "#e9a03b"), (sumS, "sum", "k")]:
        ax[r, 5].plot(rk, prof(m) / pt, color=cl, lw=2, label=t)
    ax[r, 5].axhline(1, color="0.6", lw=0.6, ls=":"); ax[r, 5].set_xscale("log"); ax[r, 5].set_ylim(0, 1.15); ax[r, 5].set_xlabel("R [kpc]")
    ax[r, 5].set_ylabel("fraction of original total")
    if r == 0: ax[r, 5].legend(fontsize=9); ax[r, 5].set_title("radial fractions", fontsize=10)
fig.colorbar(im, ax=ax[:, :2], shrink=0.4, label="log10 Σ [Msun/pMpc²]")
fig.suptitle("COSMA test: particle-binned DM + gas + stars vs the original total map (CDM, z = 0, all smoothed 1.5 px)", fontsize=14)
fig.savefig(f"{HERE}/figures/baryon_test_check.png", dpi=60)
for rw in rows: print("fof %d proj %d: median log(sum/orig) %+.3f, rms %.3f, gas frac R<0.5Mpc %.3f" % rw)

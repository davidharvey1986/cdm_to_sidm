"""Quick check: gas = total - stars - dark matter for a few BAHAMAS stamps (CDM and SIDM1).

total, stars : bahamas.pkl channels 0 and 2 (IDL SPH-projected, |los| < 5 pMpc), row idx_combined_<model>
dark matter  : velmaps_v2_imaged W0_deep5 (plain binning, |los| < 5 pMpc), stamp orientation as prepare_dm_pairs.py
All in Msun/pMpc^2, linear. Gas is computed after smoothing every map by the same Gaussian (SM px), to soften the
kernel mismatch between the SPH-projected IDL maps and the binned dark matter.
Output figures/gas_maps_check.png
usage: python3 gas_maps_check.py
"""
import os, json, pickle
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as G
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"; V = f"{IMG}/velmaps_v2_imaged"
SNAP = {0.0: "032", 0.125: "031", 0.25: "030", 0.375: "029"}
MOD = {"cdm": "CDM+baryons", "sidm1": "SIDM1+baryons"}
SM = 1.5
O = json.load(open(f"{HERE}/../velmaps/bahamas/orientation.json"))
meta, stamps = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb"))
_, full = pickle.load(open(f"{IMG}/bahamas.pkl", "rb"))
keep = np.where(meta["mask"][:, 0])[0]


def dm_raw(k, key):
    z, P = float(meta["redshift"][k]), int(meta["projection"][k]); fof = int(meta[f"fof_{key}"][k])
    o = O[f"projection_{P}"]; d = np.load(f"{V}/{MOD[key]}/snap_{SNAP[z]}/los{o['los']}.npz")
    i = int(np.where(d["fof"] == fof)[0][0]); m = np.rot90(d["moments"][i, 0].astype(np.float64), o["rot"])
    return m[:, ::-1] if o["flip"] else m


rng = np.random.default_rng(3)
full4 = keep[meta["n_models"][keep] == 4]
ks = [keep[1951]] + list(rng.choice(full4, 4, replace=False))
rows = [(k, key) for k in ks for key in ("cdm",)] + [(keep[1951], "sidm1")]
Y, X = np.indices((100, 100)); R = np.hypot(X - 49.5, Y - 49.5).astype(int)
fig, ax = plt.subplots(len(rows), 6, figsize=(24, 4 * len(rows)), constrained_layout=True)
for r, (k, key) in enumerate(rows):
    j = int(meta[f"idx_combined_{key}"][k]); tot, star = full[j][..., 0].astype(np.float64), full[j][..., 2].astype(np.float64)
    ch = list(meta["channels"]).index(MOD[key]); assert np.allclose(tot, stamps[k, ch], rtol=1e-4, equal_nan=True), "total != matched stamp"
    dm = dm_raw(k, key); tS, sS, dS = G(tot, SM), G(star, SM), G(dm, SM); gas = tS - sS - dS
    lab = f"{key.upper()} z{meta['redshift'][k]:.3f} logM {meta['log10_M200'][k]:.2f}"
    vmin, vmax = 12.5, 15.8
    for c, (m, t) in enumerate([(tS, "total"), (dS, "dark matter (binned)"), (sS, "stars"), (np.where(gas > 0, gas, np.nan), "gas = tot − stars − DM")]):
        im = ax[r, c].imshow(np.log10(np.maximum(m, 1e10)) if c < 3 else np.log10(m), origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax[r, c].set_title(t if r == 0 else ""); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[r, 0].set_ylabel(lab, fontsize=11)
    fneg = (gas < 0).mean()
    ax[r, 4].imshow(np.sign(gas) * np.minimum(np.abs(gas) / tS, 0.5), origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax[r, 4].contour(gas < 0, [0.5], colors="k", linewidths=0.6); ax[r, 4].set_xticks([]); ax[r, 4].set_yticks([])
    ax[r, 4].set_title(("gas / total (black = negative)\n" if r == 0 else "") + f"{100 * fneg:.1f}% pixels negative", fontsize=10)
    prof = lambda m: np.bincount(R.ravel(), m.ravel())[:50] / np.bincount(R.ravel())[:50]
    rk = (np.arange(50) + 0.5) * 20
    for m, t, cl in [(dS, "DM", "#3a6fb0"), (sS, "stars", "#e9a03b"), (gas, "gas", "#2a9d8f")]:
        ax[r, 5].plot(rk, prof(m) / prof(tS), color=cl, lw=2, label=t)
    ax[r, 5].axhline(0, color="0.5", lw=0.6); ax[r, 5].axhline(0.16, color="0.7", lw=0.6, ls=":")
    ax[r, 5].set_xscale("log"); ax[r, 5].set_ylim(-0.1, 1.05); ax[r, 5].set_xlabel("R [kpc]"); ax[r, 5].set_ylabel("fraction of total")
    if r == 0: ax[r, 5].legend(); ax[r, 5].set_title("radial fractions (dotted: cosmic baryon fraction 0.16)", fontsize=10)
fig.colorbar(im, ax=ax[:, :4], shrink=0.4, label="log10 Σ [Msun/pMpc²]")
fig.suptitle(f"BAHAMAS: gas = total − stars − dark matter, all maps smoothed {SM} px; row 1 = merger z0.250_291, last row = its SIDM1", fontsize=14)
fig.savefig(f"{HERE}/figures/gas_maps_check.png", dpi=60); print("wrote figures/gas_maps_check.png")

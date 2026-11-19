"""Projected halo shape eps(R) = 1 - b/a in BAHAMAS and darkskies, CDM and every SIDM model (simulations only, CPU).

  BAHAMAS DM    : data/pairs_dm.npz  (dark matter, |los| < 5 pMpc, smoothed 1.5 px)
  BAHAMAS total : data/pairs.npz     (total mass, |los| < 5 pMpc), smoothed 1.5 px here
  darkskies     : data/darkskies_pairs.npz (total mass of the hydro zooms, projected through 400 cMpc), mean background
                  (background_sigma, a function of snapshot only) subtracted, smoothed 1.5 px here
  darkskies DM  : darkskies_velmaps W0_deep5 (dark matter, |los| < 5 pMpc, binned, smoothed 1.5 px as BAHAMAS DM), every
                  halo_database row (matched, SIDM exists, not unfixable) whose CDM and SIDM velocity maps both exist
All pairs with |shift| <= 2 px and both maps finite, all splits; central 96x96 px; halo_shape.shapes.
Figure figures/shape_suites.png: mean eps(R) per model for each suite, and the paired shape change eps_SIDM - eps_CDM(R)
for both total-mass suites on one sigma colour scale.
Per-map shapes are cached in runs/shape_suites_cache.npz (delete it to re-measure); --mmin/--mmax cut on the CDM
halo's log10 M200 (pairs keep their CDM halo's mass) -> figures/shape_suites_m<mmin>-<mmax>.png
usage: python shape_suites.py [--mmin 14.7] [--mmax 16]
"""
import json, os, argparse, pickle
import pandas as pd
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from halo_shape import RK, shapes
HERE = "/Users/davidharvey/Work/cdm_to_sidm/unet"
BG = {0.375: 3.0040104973e13, 0.25: 2.4826533036e13, 0.125: 2.0109491759e13, 0.0: 1.5888981143e13}  # darkskies background_sigma
SM = lambda S: gaussian_filter(S, (0, 1.5, 1.5), mode="nearest")
ap = argparse.ArgumentParser(); ap.add_argument("--mmin", type=float, default=None); ap.add_argument("--mmax", type=float, default=None); a_ = ap.parse_args()
CACHE = f"{HERE}/runs/shape_suites_cache.npz"


def suite(file, mode):
    D = np.load(f"{HERE}/data/{file}", allow_pickle=True)
    x, y, xi, sig, sh, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "redshift"))
    ok = (np.abs(sh).max(1) <= 2) & (sig > 0) & np.isfinite(y).all((1, 2)) & np.isfinite(x[xi]).all((1, 2))
    if "ok" in D.files: ok &= D["ok"]
    P = np.where(ok)[0]; rows = np.unique(xi[P])
    def lin(L, z):
        S = 10 ** L[:, 2:98, 2:98].astype(np.float64)
        if mode == "ds": S = np.maximum(S - np.array([BG[float(q)] for q in z])[:, None, None], 0)
        return S if mode == "dm" else SM(S)
    eX = np.full((len(x), len(RK)), np.nan)
    eX[rows] = shapes(lin(x[rows], zz[rows]), lin=True); eY = shapes(lin(y[P], zz[xi[P]]), lin=True)
    print(file, mode, len(rows), "CDM maps,", len(P), "pairs", flush=True)
    return dict(eC=eX[rows], eY=eY, dE=eY - eX[xi[P]], sig=sig[P], logM=D["log10_M200"][xi[P]], logM_c=D["log10_M200"][rows])


def suite_ds_dm():
    """darkskies dark matter from the velocity maps (W0_deep5), paired by halo_database row"""
    db = pd.read_csv(f"{HERE}/../halo_database.csv"); bad = set("/".join(l.split()[:3]) for l in open(f"{HERE}/../unfixable6.txt") if l.strip().startswith("SIDM"))
    db = db[db.matched & db.sidm_exists]; db = db[~db.apply(lambda r: f"{r.sidm_model}/{r.snap}/{r.halo}" in bad, axis=1)]
    vp = lambda q: q.replace("/darkskies/", "/darkskies_velmaps/")
    db = db[[os.path.isfile(vp(q)) and os.path.isfile(vp(c)) for q, c in zip(db.sidm_path, db.cdm_path)]].reset_index(drop=True)
    W0 = lambda q: SM(np.asarray(pickle.load(open(vp(q), "rb"))[0])[4][None, 2:98, 2:98].astype(np.float64))[0]
    cp = sorted(db.cdm_path.unique()); ci = {c: i for i, c in enumerate(cp)}
    eC = shapes(np.stack([W0(c) for c in cp]), lin=True); eY = shapes(np.stack([W0(q) for q in db.sidm_path]), lin=True)
    lmc = db.groupby("cdm_path").M200c_cdm.first().reindex(cp).values
    j = np.array([ci[c] for c in db.cdm_path]); print("darkskies DM", len(cp), "CDM maps,", len(db), "pairs", flush=True)
    return dict(eC=eC, eY=eY, dE=eY - eC[j], sig=db.sigma.values.astype(float), logM=np.log10(db.M200c_cdm.values * 1e10), logM_c=np.log10(lmc * 1e10))


SUITES = {"BAHAMAS DM": ("pairs_dm.npz", "dm"), "darkskies DM": None, "BAHAMAS total": ("pairs.npz", "tot"), "darkskies total": ("darkskies_pairs.npz", "ds")}
Cc = dict(np.load(CACHE)) if os.path.isfile(CACHE) else {}; R = {}
for n, v in SUITES.items():
    if f"{n}__eC" in Cc: R[n] = {q: Cc[f"{n}__{q}"] for q in ("eC", "eY", "dE", "sig", "logM", "logM_c")}
    else: R[n] = suite(*v) if v else suite_ds_dm(); Cc.update({f"{n}__{q}": w for q, w in R[n].items()}); np.savez(CACHE, **Cc)
lo, hi = a_.mmin if a_.mmin is not None else -np.inf, a_.mmax if a_.mmax is not None else np.inf
for n, r in R.items():
    kc, kp = (r["logM_c"] > lo) & (r["logM_c"] <= hi), (r["logM"] > lo) & (r["logM"] <= hi)
    R[n] = dict(eC=r["eC"][kc], logM_c=r["logM_c"][kc], **{q: r[q][kp] for q in ("eY", "dE", "sig", "logM")})
TAG = "" if (a_.mmin is None and a_.mmax is None) else f"_m{a_.mmin if a_.mmin is not None else ''}-{a_.mmax if a_.mmax is not None else ''}"
MLAB = "" if not TAG else f"; log10 M200 in ({lo:g}, {hi:g}]"
allsig = sorted(set(np.concatenate([np.unique(r["sig"]) for r in R.values()])))
cmap = plt.cm.Blues; col = lambda s: cmap(0.35 + 0.65 * (np.log10(s) - np.log10(min(allsig))) / (np.log10(max(allsig)) - np.log10(min(allsig))))
m_se = lambda e: (np.nanmean(e, 0), np.nanstd(e, 0) / np.sqrt(np.isfinite(e).sum(0)))
fig, AX = plt.subplots(2, 3, figsize=(22, 12.5), constrained_layout=True); summ = {}
POS = {"BAHAMAS DM": (0, 0), "darkskies DM": (0, 1), "BAHAMAS total": (1, 0), "darkskies total": (1, 1)}
for name, r in R.items():
    i, jc = POS[name]; A = AX[i, jc]; D3 = AX[i, 2]; m, e = m_se(r["eC"]); A.plot(RK, m, color="k", lw=2.6, label=f"CDM ({len(r['eC'])} maps, median logM {np.median(r['logM_c']):.2f})")
    A.fill_between(RK, m - e, m + e, color="k", alpha=0.15, lw=0); summ[name] = {"CDM": m.tolist()}
    for s in np.unique(r["sig"]):
        k = r["sig"] == s; m, e = m_se(r["eY"][k]); A.plot(RK, m, color=col(s), lw=2.2, label=f"SIDM{s:g} ({k.sum()})")
        A.fill_between(RK, m - e, m + e, color=col(s), alpha=0.2, lw=0); summ[name][f"SIDM{s:g}"] = m.tolist()
        md, ed = m_se(r["dE"][k]); ls = "-" if name.startswith("BAHAMAS") else "--"
        D3.plot(RK, md, color=col(s), lw=2.2, ls=ls, marker="o" if ls == "--" else None, ms=4, label=f"{name.split()[0]} SIDM{s:g} ({k.sum()})")
        D3.fill_between(RK, md - ed, md + ed, color=col(s), alpha=0.15, lw=0); summ[name][f"dSIDM{s:g}"] = md.tolist()
    A.set_title(name, fontsize=14); A.set_ylim(0, 0.5); A.set_ylabel("mean ε = 1 − b/a within R")
for i, t in enumerate(("dark matter", "total mass")):
    AX[i, 2].axhline(0, color="0.5", lw=0.8); AX[i, 2].set_ylabel("mean ε_SIDM − ε_CDM (paired)")
    AX[i, 2].set_title(f"shape change, {t}: BAHAMAS (solid) vs darkskies (dashed)", fontsize=13)
for A in AX.ravel(): A.set_xscale("log"); A.set_xlabel("R (semi-major) [kpc]"); A.legend(fontsize=9)
fig.suptitle("Projected halo shapes, simulations only. Top: dark matter (|los| < 5 pMpc). Bottom: total mass (darkskies: 400 cMpc, background subtracted). All smoothed 30 kpc" + MLAB, fontsize=14)
fig.savefig(f"{HERE}/figures/shape_suites{TAG}.png", dpi=65); json.dump(dict(R_kpc=RK.tolist(), **summ), open(f"{HERE}/runs/shape_suites_summary{TAG}.json", "w"), indent=1)
print("R [kpc]                 ", " ".join(f"{v:6d}" for v in RK))
for name, d in summ.items():
    for k, v in d.items(): print(f"{name:16s} {k:9s}", " ".join(f"{q:+6.3f}" if k.startswith("d") else f"{q:6.3f}" for q in v))
print(f"wrote figures/shape_suites{TAG}.png")

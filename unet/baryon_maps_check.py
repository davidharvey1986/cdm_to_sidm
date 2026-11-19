"""Check the particle-binned baryon maps (COSMA bahamas_baryon_maps.py) for every matched stamp of every model:
does binned DM (velmaps_v2 W0_deep5) + gas + stars (deep5) reproduce the original IDL total map (bahamas_matched.pkl)?
Per stamp, after smoothing all maps by 1.5 px: median and rms of log10(sum / original) within R < 45 px (0.9 Mpc),
plus the mean radial profiles of sum/original and of the gas and star fractions per model.
Output figures/baryon_maps_check.png, runs/baryon_maps_check.npz
usage: python baryon_maps_check.py
"""
import os, json, pickle
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as G
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"
V, BM = f"{IMG}/velmaps_v2_imaged", f"{IMG}/baryon_maps"
O = json.load(open(f"{HERE}/../velmaps/bahamas/orientation.json"))
SNAP = {0.0: "032", 0.125: "031", 0.25: "030", 0.375: "029"}
meta, stamps = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb"))
MODELS = list(meta["channels"]); KEY = {m: m.split("+")[0].lower() for m in MODELS}
SM = 1.5; Y, X = np.indices((100, 100)); R = np.hypot(X - 49.5, Y - 49.5); RI = R.astype(int); IN = R < 45
nR = np.bincount(RI.ravel())[:50]
prof = lambda m: np.bincount(RI.ravel(), m.ravel())[:50] / nR
orient = lambda m, o: (lambda r: r[:, ::-1] if o["flip"] else r)(np.rot90(m, o["rot"]))
res = {}
for ch, mod in enumerate(MODELS):
    key = KEY[mod]; ks = np.where(meta["mask"][:, ch])[0]
    med, rms, lm, zz = (np.full(len(ks), np.nan) for _ in range(4)); P = np.zeros((3, 50)); npf = 0; miss = 0
    grp = {}
    for i, k in enumerate(ks):
        o = O[f"projection_{int(meta['projection'][k])}"]; grp.setdefault((SNAP[float(meta["redshift"][k])], o["los"]), []).append(i)
    for (snap, los), ii in sorted(grp.items()):
        dv = np.load(f"{V}/{mod}/snap_{snap}/los{los}.npz"); db = np.load(f"{BM}/{mod}/snap_{snap}/los{los}.npz")
        fv = {int(f): j for j, f in enumerate(dv["fof"])}; fb = {int(f): j for j, f in enumerate(db["fof"])}
        mv = dv["moments"]; mb = db["maps"]
        for i in ii:
            k = ks[i]; fof = int(meta[f"fof_{key}"][k]); o = O[f"projection_{int(meta['projection'][k])}"]
            if fof not in fv or fof not in fb: miss += 1; continue
            dm = orient(mv[fv[fof], 0].astype(np.float64), o)
            gas = orient(mb[fb[fof], 0].astype(np.float64), o); st = orient(mb[fb[fof], 2].astype(np.float64), o)
            tS, sumS, gS, sS = (G(m, SM) for m in (stamps[k, ch].astype(np.float64), dm + gas + st, gas, st))
            lr = np.log10(sumS / tS)[IN]; med[i], rms[i] = np.median(lr), lr.std()
            lm[i], zz[i] = meta["log10_M200"][k], meta["redshift"][k]
            pt = prof(tS); P += [prof(sumS) / pt, prof(gS) / pt, prof(sS) / pt]; npf += 1
        print(f"  {mod} snap {snap} los {los}: {len(ii)} stamps", flush=True)
    res[mod] = dict(med=med, rms=rms, lm=lm, z=zz, P=P / npf, miss=miss)
    ok = np.isfinite(med)
    print(f"{mod}: {ok.sum()} stamps checked, {miss} not found; median log(sum/orig) {np.median(med[ok]):+.4f} "
          f"(5-95%: {np.percentile(med[ok], 5):+.4f} .. {np.percentile(med[ok], 95):+.4f}); "
          f"median rms {np.median(rms[ok]):.4f}; worst |median| {np.abs(med[ok]).max():.3f}; "
          f"gas frac R<0.5Mpc {res[mod]['P'][1][:25].mean():.3f}, stars {res[mod]['P'][2][:25].mean():.3f}", flush=True)
np.savez(f"{HERE}/runs/baryon_maps_check.npz", **{f"{m}__{q}": res[m][q] for m in res for q in ("med", "rms", "lm", "z", "P")})

COL = dict(zip(MODELS, ["#222222", "#3a6fb0", "#2a9d8f", "#d1495b"])); rk = (np.arange(50) + 0.5) * 20
fig, ax = plt.subplots(1, 5, figsize=(26, 5), constrained_layout=True)
for mod in MODELS:
    r = res[mod]; ok = np.isfinite(r["med"]); c = COL[mod]; lab = mod.replace("+baryons", "")
    ax[0].hist(r["med"][ok], np.linspace(-0.03, 0.03, 61), histtype="step", lw=2, color=c, label=f"{lab} ({ok.sum()})")
    ax[1].hist(r["rms"][ok], np.linspace(0, 0.03, 61), histtype="step", lw=2, color=c, label=lab)
    ax[2].scatter(r["lm"][ok], r["med"][ok], s=3, color=c, alpha=0.3, label=lab)
    ax[3].plot(rk, r["P"][0], color=c, lw=2, label=lab)
    ax[4].plot(rk, r["P"][1], color=c, lw=2, label=f"{lab} gas"); ax[4].plot(rk, r["P"][2], color=c, lw=2, ls="--")
ax[0].set_xlabel("median log10(sum / original), R < 0.9 Mpc"); ax[0].set_ylabel("stamps")
ax[1].set_xlabel("rms of log10(sum / original), R < 0.9 Mpc")
ax[2].set_xlabel("log10 M200"); ax[2].set_ylabel("median log10(sum / original)"); ax[2].set_ylim(-0.03, 0.03)
ax[3].axhline(1, color="0.6", lw=0.6, ls=":"); ax[3].set_ylabel("mean (DM + gas + stars) / original")
ax[4].set_ylabel("mean fraction of original total (solid gas, dashed stars)")
for x in ax[3:]: x.set_xscale("log"); x.set_xlabel("R [kpc]")
for x in ax: x.legend(fontsize=9)
ax[0].set_title("per-stamp offset"); ax[1].set_title("per-stamp pixel scatter"); ax[2].set_title("offset vs mass")
ax[3].set_title("radial closure"); ax[4].set_title("baryon fractions")
fig.suptitle("All matched stamps: particle-binned DM + gas + stars vs the original total map (smoothed 1.5 px)", fontsize=14)
fig.savefig(f"{HERE}/figures/baryon_maps_check.png", dpi=70); print("wrote figures/baryon_maps_check.png")

"""Fourier coherence between each model and CDM+baryons (log10 total-mass stamps).

Per aligned pair: log maps, model rolled by the integer cross-correlation shift
(as in unet/prepare_data.py), mean subtracted, mirror-padded to 200x200.
Pooled over all pairs, per |k| bin:
  coherence C(k) = |sum F_m F_c*|^2 / (sum |F_m|^2 sum |F_c|^2)   1 = same map, 0 = unrelated
  transfer  T(k) = Re sum F_m F_c* / sum |F_c|^2                  the Fourier-filter baseline
  power ratio    = sum |F_m|^2 / sum |F_c|^2
CDM_low / CDM_hi are the no-SIDM controls: where SIDM's coherence matches theirs,
the differences are run-to-run (feedback / stochastic), not SIDM.
Output: ../figures/coherence.png
"""
import os, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"
MODELS = ["CDM_low+baryons", "CDM_hi+baryons", "SIDM0.1+baryons", "SIDM0.3+baryons", "SIDM1+baryons"]
NPAD = 200
fy, fx = np.meshgrid(np.fft.fftfreq(NPAD) * NPAD, np.fft.fftfreq(NPAD) * NPAD, indexing="ij")
KB = np.rint(np.hypot(fy, fx)).astype(int)
NKB = KB.max() + 1


def best_shift(a, b, maxs=12):
    a, b = a - a.mean(), b - b.mean()
    xc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    d = (np.arange(100) + 50) % 100 - 50
    ok = np.abs(d) <= maxs
    xc = np.where(ok[:, None] & ok[None, :], xc, -np.inf)
    i, j = np.unravel_index(np.argmax(xc), xc.shape)
    return int(d[i]), int(d[j])


load = lambda f: pickle.load(open(f"{IMG}/{f}", "rb"))[1]
db = pd.read_csv(f"{HERE}/bahamas_halo_database.csv")
cdm = load("bahamas_cdm.pkl")
out = {}
for mname in MODELS:
    d = db[(db.model == mname) & db.aligned]
    other = load(d.model_file.iloc[0])
    cross, pc, pm = (np.zeros(NKB, complex), np.zeros(NKB), np.zeros(NKB))
    n = 0
    for ic, im_ in zip(d.idx_model_file_cdm, d.idx_model_file_model):
        C, M = np.asarray(cdm[ic])[..., 0].astype(float), np.asarray(other[im_])[..., 0].astype(float)
        if (C <= 0).any() or (M <= 0).any():
            continue
        lC, lM = np.log10(C), np.log10(M)
        lM = np.roll(lM, best_shift(lC, lM), (0, 1))
        Fc = np.fft.fft2(np.pad(lC - lC.mean(), ((0, 100), (0, 100)), mode="symmetric"))
        Fm = np.fft.fft2(np.pad(lM - lM.mean(), ((0, 100), (0, 100)), mode="symmetric"))
        cross += np.bincount(KB.ravel(), (Fm * Fc.conj()).real.ravel(), NKB) \
            + 1j * np.bincount(KB.ravel(), (Fm * Fc.conj()).imag.ravel(), NKB)
        pc += np.bincount(KB.ravel(), (np.abs(Fc) ** 2).ravel(), NKB)
        pm += np.bincount(KB.ravel(), (np.abs(Fm) ** 2).ravel(), NKB)
        n += 1
    out[mname] = dict(n=n, coh=np.abs(cross) ** 2 / (pc * pm), T=cross.real / pc, pr=pm / pc)

k = np.arange(NKB)
lam = NPAD / np.maximum(k, 1)  # wavelength in original pixels
sel = (k >= 2) & (k <= 100)  # 100 px down to the 2 px Nyquist
print(f"{'model':18s} {'n':>5}   coherence at wavelength  50/20/10/5/3 px      T at 50/20/10/5/3 px")
for mname, o in out.items():
    pick = [int(round(NPAD / l)) for l in (50, 20, 10, 5, 3)]
    print(f"{mname:18s} {o['n']:5d}   " + " ".join(f"{o['coh'][p]:.3f}" for p in pick)
          + "     " + " ".join(f"{o['T'][p]:.3f}" for p in pick))
lam_half = {m: lam[sel][np.argmax(o["coh"][sel] < 0.5)] for m, o in out.items()}
print("wavelength where coherence drops below 0.5 [px]:", {m.replace('+baryons', ''): round(float(v), 1) for m, v in lam_half.items()})

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"CDM_low+baryons": "#8a8983", "CDM_hi+baryons": "#4a4a46",
       "SIDM0.1+baryons": "#9ec5f0", "SIDM0.3+baryons": "#2a78d6", "SIDM1+baryons": "#0d3f80"}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), facecolor="white")
for mname, o in out.items():
    kw = dict(color=COL[mname], lw=2, ls="--" if mname.startswith("CDM") else "-", label=mname.replace("+baryons", ""))
    ax[0].plot(lam[sel], o["coh"][sel], **kw)
    ax[1].plot(lam[sel], o["T"][sel], **kw)
    ax[2].plot(lam[sel], o["pr"][sel], **kw)
for a, t in zip(ax, ("coherence with CDM", "transfer function T(k) (Fourier filter)", "power ratio P_model / P_CDM")):
    a.set_xscale("log"); a.invert_xaxis(); a.set_xlabel("wavelength [px]  (small scales to the right)")
    a.set_title(t, fontsize=10); a.axhline(1 if "coherence" not in t else 0.5, color="#6b6a63", lw=0.6)
ax[0].set_ylim(0, 1.02); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
f = f"{HERE}/../figures/coherence.png"
fig.savefig(f, dpi=130)
print(f)

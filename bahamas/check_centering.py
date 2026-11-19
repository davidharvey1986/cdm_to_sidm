"""Is the central dip in log10(SIDM/CDM) a centring artefact?

For every aligned pair (model vs CDM+baryons, total-mass channel) compare the
radial profile of the log ratio measured two ways:
  integer : as in unet/prepare_data.py - roll the model map by the integer
            shift that maximises the FFT cross-correlation, then average
            log(model) - log(CDM) in 1-px annuli about the stamp centre
  subpix  : profile each map about its OWN sub-pixel density centre
            (iterated Sigma-weighted centroid, r < 3 px, from the peak pixel),
            then difference the two profiles
Controls: CDM_low and CDM_hi (no SIDM) - any dip they show is not SIDM.
Also a pure self-test: CDM shifted by a sub-pixel amount (Fourier shift) vs itself.
Output: ../figures/centering_check.png and a printed table.
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy.ndimage import fourier_shift

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"
MODELS = ["CDM_low+baryons", "CDM_hi+baryons", "SIDM0.1+baryons", "SIDM0.3+baryons", "SIDM1+baryons"]
NR = 30
yy, xx = np.indices((100, 100))


def chan0(f):
    meta, im = pickle.load(open(f"{IMG}/{f}", "rb"))
    return im  # list/array of (100,100,3)


def best_shift(a, b, maxs=12):
    a, b = a - a.mean(), b - b.mean()
    xc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    d = (np.arange(100) + 50) % 100 - 50
    ok = np.abs(d) <= maxs
    xc = np.where(ok[:, None] & ok[None, :], xc, -np.inf)
    i, j = np.unravel_index(np.argmax(xc), xc.shape)
    return int(d[i]), int(d[j])


def centre(S, r0=3.0, it=10):
    """Sub-pixel centre: start at the max pixel within 15 px of the middle, iterate a weighted centroid."""
    inner = np.hypot(yy - 49.5, xx - 49.5) < 15
    cy, cx = np.unravel_index(np.argmax(np.where(inner, S, -np.inf)), S.shape)
    cy, cx = float(cy), float(cx)
    for _ in range(it):
        w = S * (np.hypot(yy - cy, xx - cx) < r0)
        cy, cx = (w * yy).sum() / w.sum(), (w * xx).sum() / w.sum()
    return cy, cx


def prof(L, cy, cx):
    rb = np.hypot(yy - cy, xx - cx).astype(int)
    m = rb < NR
    return np.bincount(rb[m], L[m], NR) / np.maximum(np.bincount(rb[m], minlength=NR), 1)


db = pd.read_csv(f"{HERE}/bahamas_halo_database.csv")
cdm = chan0("bahamas_cdm.pkl")
res, offs = {}, {}
for mname in MODELS:
    d = db[(db.model == mname) & db.aligned]
    other = chan0(d.model_file.iloc[0])
    P_int, P_sub, off_c, off_rel = [], [], [], []
    for ic, im_ in zip(d.idx_model_file_cdm, d.idx_model_file_model):
        C, M = np.asarray(cdm[ic])[..., 0].astype(float), np.asarray(other[im_])[..., 0].astype(float)
        if (C <= 0).any() or (M <= 0).any():
            continue
        lC, lM = np.log10(C), np.log10(M)
        dy, dx = best_shift(lC, lM)
        P_int.append(prof(np.roll(lM, (dy, dx), (0, 1)) - lC, 49.5, 49.5))
        c1, c2 = centre(C), centre(M)
        P_sub.append(prof(lM, *c2) - prof(lC, *c1))
        off_c.append(np.hypot(c1[0] - 49.5, c1[1] - 49.5))
        off_rel.append(np.hypot(c2[0] + dy - c1[0], c2[1] + dx - c1[1]))  # residual after integer roll
    res[mname] = np.array(P_int), np.array(P_sub)
    offs[mname] = np.array(off_c), np.array(off_rel)

# self-test: CDM vs CDM shifted by a random sub-pixel offset (|dy|,|dx| <= 0.5 px)
rng = np.random.default_rng(0)
d = db[(db.model == MODELS[0]) & db.aligned]
P_self = []
for ic in d.idx_model_file_cdm.values[:1000]:
    C = np.asarray(cdm[ic])[..., 0].astype(float)
    sh = rng.uniform(-0.5, 0.5, 2)
    Cs = np.clip(np.fft.ifft2(fourier_shift(np.fft.fft2(C), sh)).real, C.min() * 0.5, None)
    P_self.append(prof(np.log10(Cs) - np.log10(C), 49.5, 49.5))
P_self = np.array(P_self)

print(f"{'model':18s} {'n':>5}  {'r=0 int':>8} {'r=0 sub':>8}  {'r<4 int':>8} {'r<4 sub':>8}  "
      f"{'CDM ctr off':>11} {'resid off':>9}  (medians; dex, px)")
for mname in MODELS:
    Pi, Ps = res[mname]; oc, orl = offs[mname]
    print(f"{mname:18s} {len(Pi):5d}  {np.median(Pi[:, 0]):+8.3f} {np.median(Ps[:, 0]):+8.3f}  "
          f"{np.median(Pi[:, :4].mean(1)):+8.3f} {np.median(Ps[:, :4].mean(1)):+8.3f}  "
          f"{np.median(oc):11.2f} {np.median(orl):9.2f}")
print(f"{'self, <=0.5px':18s} {len(P_self):5d}  {np.median(P_self[:, 0]):+8.3f}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED = "#1f1f1e", "#6b6a63"
COL = {"CDM_low+baryons": "#8a8983", "CDM_hi+baryons": "#4a4a46",
       "SIDM0.1+baryons": "#9ec5f0", "SIDM0.3+baryons": "#2a78d6", "SIDM1+baryons": "#0d3f80"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white", sharey=True)
r = np.arange(NR) + 0.5
for a, k, t in ((ax[0], 0, "integer re-centring (as in training)"), (ax[1], 1, "each map about its own sub-pixel centre")):
    for mname in MODELS:
        a.plot(r, np.median(res[mname][k], 0), color=COL[mname], lw=2,
               ls="--" if mname.startswith("CDM") else "-", label=mname.replace("+baryons", ""))
    if k == 0:
        a.plot(r, np.median(P_self, 0), color="#eb6834", lw=1.5, ls=":", label="CDM vs itself, shifted <=0.5 px")
    a.axhline(0, color=MUTED, lw=0.6); a.set_xlim(0, 25); a.set_title(t, fontsize=10, color=INK)
    a.set_xlabel("radius [px]")
ax[0].set_ylabel("median log10(model / CDM) [dex]")
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
out = f"{HERE}/../figures/centering_check.png"
fig.savefig(out, dpi=130)
print(out)

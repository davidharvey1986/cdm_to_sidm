"""Stacked density profiles of subhaloes in CDM and SIDM1 maps (BAHAMAS, all clusters with an SIDM1 pair).
Subhaloes are found independently in each map (positions are not shared between runs): local maxima of the
high-passed log map (> 0.15 dex, r > 6 px from the cluster centre, 8 px from the border; same as peak_align.py).
For each: 17x17 px cutout of linear Sigma, host level = mean in the 6-8 px annulus around the peak,
excess E(r) = azimuthal mean Sigma(r) - host. Subhalo mass proxy = excess mass inside 3 px.
Profiles are stacked in bins of that mass, so CDM and SIDM1 are compared at fixed subhalo mass.
usage: python3 subhalo_profiles.py   -> figures/subhalo_profiles.png
"""
import os
import numpy as np, matplotlib
from scipy.ndimage import gaussian_filter, maximum_filter
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig, sh = (D[k] for k in ("x", "y", "xi", "sigma", "shift"))
H = 8; cy, cx = np.indices((2 * H + 1, 2 * H + 1)) - H; cr = np.hypot(cy, cx); crb = np.minimum(np.round(cr).astype(int), H)
cnt = np.bincount(crb.ravel()); ANN = (cr >= 6) & (cr <= 8); IN3 = cr < 3
n = 100; yy, xx = np.indices((n, n)); r = np.hypot(yy - 49.5, xx - 49.5)
PIX_KPC = 20.0; AREA = (PIX_KPC / 1e3) ** 2  # pMpc^2 per pixel; Sigma in Msun/pMpc^2


def subhaloes(m):
    hp = m - gaussian_filter(m, 3.0)
    pk = (maximum_filter(hp, 5) == hp) & (hp > 0.15) & (r > 6)
    pk[:H] = pk[-H:] = 0; pk[:, :H] = pk[:, -H:] = 0
    L = 10.0 ** m.astype(np.float64); out = []
    for i, j in zip(*np.nonzero(pk)):
        cut = L[i - H:i + H + 1, j - H:j + H + 1]; host = cut[ANN].mean()
        prof = np.bincount(crb.ravel(), cut.ravel()) / cnt - host
        out.append((np.log10(max((cut[IN3] - host).sum() * AREA, 1.0)), r[i, j], host, *prof))
    return out


k = np.where((np.abs(sh).max(1) <= 2) & (sig == 1.0))[0]
cats = {}
for nm, maps in (("CDM", x[xi[k]]), ("SIDM1", y[k])):
    rows = []
    for m in maps: rows += subhaloes(m)
    cats[nm] = np.array(rows); print(nm, len(rows), "subhaloes in", len(k), "maps")
np.savez_compressed(f"{HERE}/data/subhalo_profiles.npz", **cats)

MBINS = [11.0, 11.4, 11.8, 12.2, 13.0]
col = {"CDM": "k", "SIDM1": "#c0392b"}
fig, ax = plt.subplots(1, 4, figsize=(21, 5), constrained_layout=True)
rr = np.arange(H + 1) * PIX_KPC
for b, (lo, hi) in enumerate(zip(MBINS[:-1], MBINS[1:])):
    for nm, C in cats.items():
        s = (C[:, 0] >= lo) & (C[:, 0] < hi)
        if s.sum() < 20: continue
        P = C[s, 3:]; med = np.median(P, 0)
        ax[0].plot(rr[:7], med[:7], color=col[nm], ls=["-", "--", "-.", ":"][b], lw=2,
                   label=f"{nm}, log M(<60 kpc) {lo}-{hi} (n={s.sum()})")
        if nm == "SIDM1":
            Pc = np.median(cats["CDM"][(cats["CDM"][:, 0] >= lo) & (cats["CDM"][:, 0] < hi), 3:], 0)
            ax[1].plot(rr[:5], np.log10(med[:5] / Pc[:5]), "o-", color=["#e9a03b", "#c0392b", "#8e44ad", "#3a6fb0"][b],
                       label=f"log M {lo}-{hi}")
ax[0].set_yscale("log"); ax[0].set_xlabel("r from subhalo peak [kpc]"); ax[0].set_ylabel("excess Σ over local host [M☉/pMpc²]")
ax[0].set_title("stacked subhalo profiles at fixed mass (median)"); ax[0].legend(fontsize=6)
ax[1].axhline(0, color="0.5", lw=0.7); ax[1].set_xlabel("r [kpc]"); ax[1].set_ylabel("log10 SIDM1 / CDM")
ax[1].set_title("SIDM1 / CDM profile at fixed subhalo mass"); ax[1].legend(fontsize=7)
mb = np.linspace(10.6, 13.0, 25); mc = 0.5 * (mb[1:] + mb[:-1])
for nm, C in cats.items():
    h = np.histogram(C[:, 0], mb)[0] / len(k); ax[2].step(mc, h, where="mid", color=col[nm], lw=2, label=nm)
ax[2].set_yscale("log"); ax[2].set_xlabel("log10 subhalo excess mass inside 60 kpc [M☉]"); ax[2].set_ylabel("subhaloes per map per bin")
ax[2].legend(); ax[2].set_title("subhalo mass function")
rb_ = np.arange(6, 50, 4); rc = 0.5 * (rb_[1:] + rb_[:-1])
for nm, C in cats.items():
    h = np.histogram(C[:, 1], rb_)[0] / len(k); ax[3].plot(rc * PIX_KPC / 1e3, h, "o-", color=col[nm], label=nm)
ax[3].set_xlabel("distance from cluster centre [Mpc]"); ax[3].set_ylabel("subhaloes per map per bin"); ax[3].legend()
ax[3].set_title("radial distribution")
fig.suptitle("BAHAMAS subhaloes found independently in CDM and SIDM1 maps (20 kpc pixels)")
out = f"{HERE}/figures/subhalo_profiles.png"; fig.savefig(out, dpi=95); print(out)

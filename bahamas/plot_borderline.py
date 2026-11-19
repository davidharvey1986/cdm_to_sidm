"""Close-up of one bahamas_matched.pkl sample: two channels side by side, the
second with the first's contours, their log ratio, and the pixel shift that
best aligns them (FFT cross-correlation of the log maps).
usage: python3 plot_borderline.py [sample=3201] [chan_a=0] [chan_b=3]
Output: ../figures/bahamas_borderline_<sample>.png"""
import os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"
i, a, b = (int(x) for x in (sys.argv[1:4] + ["3201", "0", "3"][len(sys.argv) - 1:]))
meta, im = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb"))
tags = ["cdm", "sidm0.1", "sidm0.3", "sidm1"]
names = ["CDM", "SIDM0.1", "SIDM0.3", "SIDM1"]
A, B = im[i, a], im[i, b]
la, lb = np.log10(A), np.log10(B)


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())


# best integer shift of B onto A
xc = np.fft.ifft2(np.fft.fft2(la - la.mean()) * np.conj(np.fft.fft2(lb - lb.mean()))).real
dy, dx = np.unravel_index(np.argmax(xc), xc.shape)
dy, dx = (dy + 50) % 100 - 50, (dx + 50) % 100 - 50
c0 = corr(la, lb)
c1 = corr(la[max(dy, 0):100 + min(dy, 0), max(dx, 0):100 + min(dx, 0)],
          np.roll(lb, (dy, dx), (0, 1))[max(dy, 0):100 + min(dy, 0), max(dx, 0):100 + min(dx, 0)])

INK, MUTED = "#1f1f1e", "#6b6a63"
vmin, vmax = np.percentile(np.r_[A.ravel(), B.ravel()], [1, 99.9])
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="white",
                       gridspec_kw=dict(wspace=0.08))
lev = np.percentile(la, [70, 90, 97, 99.5])
for k, (m, n) in enumerate(((A, names[a]), (B, names[b]))):
    im_ = ax[k].imshow(m, origin="lower", cmap="magma", norm=LogNorm(vmin, vmax))
    sep = meta[f"sep_Mpc_h_{tags[(a, b)[k]]}"][i]
    ax[k].set_title(f"{n}   fof {meta[f'fof_{tags[(a, b)[k]]}'][i]}   "
                    f"log M200 {meta[f'log10_M200_{tags[(a, b)[k]]}'][i]:.2f}\n"
                    f"centre offset from CDM halo: {sep:.3f} Mpc/h", fontsize=9, color=INK)
    ax[k].plot(49.5, 49.5, "+", color="white", ms=12, mew=1.5)
ax[1].contour(la, levels=lev, colors="#2a78d6", linewidths=0.9, origin="lower")
ax[1].plot(49.5 - dx, 49.5 - dy, "x", color="#2a78d6", ms=10, mew=2)
r = ax[2].imshow(lb - la, origin="lower", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax[2].set_title(f"log10({names[b]} / {names[a]})", fontsize=9, color=INK)
for x in ax:
    x.set_xticks([0, 25, 50, 75, 99]); x.set_yticks([0, 25, 50, 75, 99])
    x.tick_params(labelsize=7, colors=MUTED)
fig.colorbar(im_, ax=ax[:2], shrink=0.85, pad=0.01, label="total mass map")
fig.colorbar(r, ax=ax[2], shrink=0.85, pad=0.02, label="dex")
fig.suptitle(f"sample {i}: {meta['match_id'][i]}  projection {meta['projection'][i]}   "
             f"pixel correlation {c0:.2f}; best shift of {names[b]} = ({dx:+d}, {dy:+d}) px "
             f"-> {c1:.2f}\nBlue contours / x: {names[a]} map and its centre on the {names[b]} map",
             fontsize=9.5, color=INK, y=1.04)
out = os.path.normpath(f"{HERE}/../figures/bahamas_borderline_{i}.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(out, "shift", dx, dy, "corr", round(c0, 3), "->", round(c1, 3))

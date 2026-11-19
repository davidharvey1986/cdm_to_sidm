"""2D power spectra |FFT|^2 (Hann window, fftshift) of one cluster's log-delta maps (SIDM 0.1/0.3/1 minus CDM)
and of their average; last panel: azimuthal average of each.
usage: python3 plot_merger_power2d.py [row]"""
import os, sys
import numpy as np, matplotlib
from matplotlib.colors import LogNorm
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); row = int(sys.argv[1]) if len(sys.argv) > 1 else 1951
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig = (D[k] for k in ("x", "y", "xi", "sigma"))
SIG = (0.1, 0.3, 1.0); c = x[row, 2:98, 2:98].astype(float); N = 96
maps = {f"Δ, σ = {s}": y[np.where((xi == row) & (sig == s))[0][0], 2:98, 2:98] - c for s in SIG}
maps["average of the three Δ"] = np.mean(list(maps.values()), 0)
w = np.outer(np.hanning(N), np.hanning(N))
P2 = {k: np.fft.fftshift(np.abs(np.fft.fft2((v - v.mean()) * w)) ** 2) for k, v in maps.items()}
k1 = np.fft.fftshift(np.fft.fftfreq(N)); KY, KX = np.meshgrid(k1, k1, indexing="ij"); KR = np.hypot(KY, KX)
kb = np.minimum((KR * N).astype(int), N // 2)
vmin, vmax = np.percentile(np.concatenate([p.ravel() for p in P2.values()]), [5, 99.9])
fig, ax = plt.subplots(1, 5, figsize=(23, 4.7), constrained_layout=True)
ext = [k1[0], k1[-1], k1[0], k1[-1]]
for j, (nm, p) in enumerate(P2.items()):
    im = ax[j].imshow(p, norm=LogNorm(vmin, vmax), cmap="viridis", origin="lower", extent=ext)
    ax[j].set_title(f"2D power of {nm}", fontsize=10); ax[j].set_xlabel("k_x [1/px]"); ax[j].set_ylabel("k_y [1/px]")
fig.colorbar(im, ax=ax[:4], shrink=0.85, label="|FFT|² [dex² px²]")
g = ax[4]
for (nm, p), cl in zip(P2.items(), ("#2a9d8f", "#e9a03b", "#c0392b", "k")):
    pr = np.bincount(kb.ravel(), p.ravel(), N // 2 + 1)[1:N // 2] / np.bincount(kb.ravel(), minlength=N // 2 + 1)[1:N // 2]
    g.plot(N / np.arange(1, N // 2), pr, color=cl, lw=2.2 if cl == "k" else 1.5, label=nm)
g.set_xscale("log"); g.set_yscale("log"); g.set_xlabel("wavelength [px]"); g.set_ylabel("azimuthal mean power")
g.legend(fontsize=8); g.set_title("azimuthally averaged")
fig.suptitle(f"BAHAMAS row {row} (merger): power spectra of the log-delta maps")
out = f"{HERE}/figures/merger_power2d_{row}.png"; fig.savefig(out, dpi=95); print(out)

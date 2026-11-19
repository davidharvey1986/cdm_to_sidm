"""One-point function (pixel PDF of log10 Sigma) of one cluster (default merger row 1951) in CDM, SIDM 0.1, 0.3, 1:
PDF, ratio to CDM, and the high-density tail as a cumulative N(>log Sigma).
usage: python3 plot_merger_onepoint.py [row]"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); row = int(sys.argv[1]) if len(sys.argv) > 1 else 1951
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig = (D[k] for k in ("x", "y", "xi", "sigma"))
maps = {"CDM": x[row, 2:98, 2:98].ravel()}
for s in (0.1, 0.3, 1.0):
    maps[f"SIDM {s}"] = y[np.where((xi == row) & (sig == s))[0][0], 2:98, 2:98].ravel()
col = {"CDM": "k", "SIDM 0.1": "#2a9d8f", "SIDM 0.3": "#e9a03b", "SIDM 1.0": "#c0392b"}
b = np.linspace(12.6, 15.4, 57); mid = 0.5 * (b[1:] + b[:-1])
fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
H = {k: np.histogram(v, b)[0] for k, v in maps.items()}
for k, v in maps.items():
    ax[0].step(mid, H[k] / len(v) / (b[1] - b[0]), where="mid", color=col[k], lw=2.2 if k == "CDM" else 1.6, label=k)
    if k != "CDM":
        ok = (H["CDM"] > 10)
        rat = (H[k] + 0.5) / (H["CDM"] + 0.5); err = np.sqrt(1 / (H[k] + 0.5) + 1 / (H["CDM"] + 0.5)) * rat
        ax[1].errorbar(mid[ok], rat[ok], err[ok], color=col[k], lw=1.6, capsize=0, marker="o", ms=3, label=k)
    srt = np.sort(v)[::-1]; ax[2].plot(srt, np.arange(1, len(srt) + 1), color=col[k], lw=2.2 if k == "CDM" else 1.6, label=k)
ax[0].set_yscale("log"); ax[0].set_xlabel("pixel log10 Σ [M☉/pMpc²]"); ax[0].set_ylabel("PDF"); ax[0].legend(); ax[0].set_title("one-point function")
ax[1].axhline(1, color="0.5", lw=0.7); ax[1].set_xlabel("pixel log10 Σ"); ax[1].set_ylabel("PDF ratio to CDM (Poisson errors)")
ax[1].set_title("SIDM / CDM"); ax[1].set_ylim(0, 2.5); ax[1].legend()
ax[2].set_yscale("log"); ax[2].set_xlim(14.2, 15.4); ax[2].set_xlabel("pixel log10 Σ"); ax[2].set_ylabel("number of pixels above")
ax[2].set_title("high-density tail: N(> log Σ)"); ax[2].legend()
fig.suptitle(f"BAHAMAS row {row} (merger): one-point function of the 96×96 px map")
out = f"{HERE}/figures/merger_onepoint_{row}.png"; fig.savefig(out, dpi=100); print(out)

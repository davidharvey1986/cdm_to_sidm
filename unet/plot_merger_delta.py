"""Merger (BAHAMAS row 1951 by default): log delta = log10 SIDM - log10 CDM for each SIDM model, against
  - the cluster's noise-reduced response sigma^p * b (from sigma_sequence.py: all three SIDM runs fitted together)
  - the mass+z template converter (template sigma^p b(r; M, z) + small-scale map), trained on BAHAMAS (in-suite)
    and on darkskies (cross-suite); suites are fitted separately, never combined.
usage: python3 plot_merger_delta.py [row]"""
import os, sys, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import two_scale as T2

HERE = os.path.dirname(os.path.abspath(__file__)); row = int(sys.argv[1]) if len(sys.argv) > 1 else 1951
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig, zz, M = (D[k] for k in ("x", "y", "xi", "sigma", "redshift", "log10_M200"))
SIG = (0.1, 0.3, 1.0); c = x[row, 2:98, 2:98].astype(np.float64); z, m = float(zz[row]), float(M[row])
true = {s: y[np.where((xi == row) & (sig == s))[0][0], 2:98, 2:98] - c for s in SIG}
F = np.load(f"{HERE}/runs/sigma_sequence/fit.npz"); k = np.where(F["cdm_row"] == row)[0]
pB = json.load(open(f"{HERE}/runs/two_scale_bahamas_nocnn/template.json"))["template"]["p"]
nr = {s: (s ** pB) * F["b"][k[0]] for s in SIG} if len(k) else None
MT = json.load(open(f"{HERE}/runs/mass_template/summary.json"))
conv = {}
for tag in ("bahamas", "darkskies"):
    J = json.load(open(f"{HERE}/runs/two_scale_{tag}_nocnn/template.json")); C = np.array(MT[tag]["mass+z"]["C"]); p = J["template"]["p"]
    bprof = C[0] + C[1] * (m - 14.5) + C[2] * (z - 0.2)
    for s in SIG:
        Lc = T2.smooth(c[None]); Sc = c[None] - Lc
        conv[tag, s] = (s ** p) * bprof[T2.RB] + (T2.apply_small(Sc, np.array([s]), J["small"]) - Sc)[0]
cols = [("true Δ (one SIDM run)", true)]
if nr: cols.append(("noise-reduced response σ^p·b\n(3 SIDM runs fitted jointly)", nr))
cols += [("converter, BAHAMAS-trained\n(mass+z template + small-scale)", {s: conv["bahamas", s] for s in SIG}),
         ("converter, darkskies-trained\n(cross-suite)", {s: conv["darkskies", s] for s in SIG})]
nc = len(cols) + 2
fig, ax = plt.subplots(3, nc, figsize=(3.9 * nc, 12), constrained_layout=True)
lo, hi = np.percentile(c, [1, 99.9]); kw = dict(origin="lower", interpolation="nearest")
rr = np.arange(48) + 0.5; rr[0] = 0.35
for i, s in enumerate(SIG):
    ax[i, 0].imshow(c, cmap="magma", vmin=lo, vmax=hi, **kw); ax[i, 0].set_ylabel(f"σ = {s}", fontsize=14)
    for j, (nm, Mp) in enumerate(cols, 1):
        im = ax[i, j].imshow(Mp[s], cmap="RdBu_r", vmin=-0.25, vmax=0.25, **kw)
        if i == 0: ax[i, j].set_title(nm, fontsize=10)
    g = ax[i, -1]
    for (nm, Mp), cl, ls in zip(cols, ("k", "#8e44ad", "#3a6fb0", "#c0392b"), ("-", "-", "--", ":")):
        g.plot(rr, T2.rprof(np.asarray(Mp[s])[None])[0], color=cl, ls=ls, lw=2, label=nm.split("\n")[0])
    g.axhline(0, color="0.5", lw=0.7); g.set_xscale("log"); g.set_ylim(-0.3, 0.12); g.set_xlabel("r [px]")
    if i == 0: g.legend(fontsize=7); g.set_title("azimuthal mean Δ", fontsize=10)
ax[0, 0].set_title(f"CDM (log M200 {m:.2f}, z {z})")
for a_ in ax[:, :-1].ravel(): a_.set_xticks([]); a_.set_yticks([])
fig.colorbar(im, ax=ax[:, 1:-1], shrink=0.5, label="Δ = log10 SIDM − log10 CDM [dex]")
fig.suptitle(f"BAHAMAS test row {row} (merger): log Δ for each SIDM model", fontsize=15)
out = f"{HERE}/figures/merger_delta_{row}.png"; fig.savefig(out, dpi=85); print(out)

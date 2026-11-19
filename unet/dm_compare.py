"""Compare the probabilistic DM-only converters on the BAHAMAS test split (BAHAMAS -> BAHAMAS).
Reads runs/dm_<name>/draws.npz for the names given (missing ones skipped).
Figure 1 (figures/dm_compare_stats.png), per sigma: mean Delta profile; rms Delta(r); power of Delta; calibration
(PIT: rank of the true pixel among the draws, flat = calibrated); one-point PDF of Delta.
Figure 2 (figures/dm_compare_structure.png): power of the SIDM map itself relative to CDM (graininess) and peak counts
per map (all / bright), true vs each method; and the per-pixel correlation of |Delta| with |CDM small-scale structure|.
Figure 3 (figures/dm_compare_merger.png): merger row 1951, true SIDM1 and two draws per method, maps and Delta.
usage: python3 dm_compare.py gauss displace displace_u2 flow
"""
import sys, os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dm_common as C, two_scale as T2
names = sys.argv[1:] or ["gauss", "displace", "flow"]
S, D = C.load(); te = S["test"]; M = {}
for n in names:
    f = f"{C.HERE}/runs/dm_{n}/draws.npz"
    if os.path.isfile(f):
        d = np.load(f); assert (d["k"] == te["k"]).all(); M[n] = d["draws"].astype(np.float32); print(n, M[n].shape, flush=True)
COL = dict(zip(M, ["#c0392b", "#2a9d8f", "#e9a03b", "#3a6fb0", "#8e44ad", "#d6338a"]))
RB = T2.RB; cnt = np.bincount(RB.ravel()); Rk = np.arange(48) * 20 + 10
prof = lambda m: np.bincount(RB.ravel(), m.ravel(), 48) / cnt
def pspec(X):
    X = X - X.mean((-2, -1), keepdims=True); w = np.hanning(C.N)[:, None] * np.hanning(C.N)[None]
    p = (np.abs(np.fft.fft2(X * w)) ** 2).reshape(-1, C.N, C.N).mean(0); return (np.bincount(C.KB.ravel(), p.ravel()) / np.bincount(C.KB.ravel()))[1:C.N // 2]
kx = np.arange(1, C.N // 2); summ = {}
fig, ax = plt.subplots(3, 5, figsize=(27, 15), constrained_layout=True)
fig2, ax2 = plt.subplots(3, 4, figsize=(23, 15), constrained_layout=True)
for i, s in enumerate(C.SIG):
    m = te["sigma"] == s; c, t = te["c"][m], te["s"][m]; dT = t - c; A, B = ax[i], ax2[i]; summ[str(s)] = {}
    A[0].plot(Rk, prof(dT.mean(0)), "k", lw=2.4, label="true"); A[1].plot(Rk, np.sqrt(prof((dT ** 2).mean(0))), "k", lw=2.4, label="true")
    A[2].loglog(C.N / kx, pspec(dT), "k", lw=2.4, label="true")
    bins = np.linspace(-0.8, 0.8, 81); A[4].hist(dT[:, T2.FIELD].ravel(), bins, histtype="step", density=True, color="k", lw=2.4, label="true")
    pc = pspec(c); B[0].loglog(C.N / kx, pspec(t) / pc, "k", lw=2.4, label="true")
    npk_t, nb_t = C.peaks(t); npk_c, nb_c = C.peaks(c)
    pk_rows = [("CDM", npk_c.mean(), nb_c.mean(), "0.6"), ("true", npk_t.mean(), nb_t.mean(), "k")]
    Sabs = np.abs(te["Sc"][m]); cor = lambda d_: np.corrcoef(np.abs(d_[:, T2.FIELD]).ravel(), Sabs[:, T2.FIELD].ravel())[0, 1]
    cors = [("true", cor(dT), "k")]
    for n, X in M.items():
        Xm = X[:, m]; dX = Xm - c[None]; cl = COL[n]
        A[0].plot(Rk, prof(dX.mean((0, 1))), color=cl, lw=1.6, ls="--", label=n)
        A[1].plot(Rk, np.sqrt(prof((dX[0] ** 2).mean(0))), color=cl, lw=1.6, ls="--", label=n)
        A[2].loglog(C.N / kx, pspec(dX[0]), color=cl, lw=1.6, ls="--", label=n)
        A[4].hist(dX[0][:, T2.FIELD].ravel(), bins, histtype="step", density=True, color=cl, lw=1.6, ls="--", label=n)
        rank = (Xm < t[None]).sum(0); h = np.bincount(rank[:, T2.FIELD].ravel(), minlength=len(Xm) + 1) / rank[:, T2.FIELD].size
        A[3].plot(np.arange(len(Xm) + 1), h * (len(Xm) + 1), color=cl, lw=2, marker="o", ms=3, label=n)
        B[0].loglog(C.N / kx, pspec(Xm[0]) / pc, color=cl, lw=1.6, ls="--", label=n)
        npk, nb = C.peaks(Xm[0]); pk_rows.append((n, npk.mean(), nb.mean(), cl)); cors.append((n, cor(dX[0]), cl))
        cov = lambda q: float(np.mean((rank[:, T2.FIELD] >= q * (len(Xm) + 1) - 0.5) & (rank[:, T2.FIELD] <= (1 - q) * (len(Xm) + 1) - 0.5)))
        summ[str(s)][n] = dict(core_mean=float(dX.mean((0, 1))[T2.CORE].mean()), rms=float(dX[0].std()), peaks=float(npk.mean()), bright=float(nb.mean()),
                               central90=cov(0.05), corr_absS=float(cors[-1][1]), pit_chi2=float(((h * (len(Xm) + 1) - 1) ** 2).sum()))
    summ[str(s)]["true"] = dict(core_mean=float(dT[:, T2.CORE].mean()), rms=float(dT.std()), peaks=float(npk_t.mean()), bright=float(nb_t.mean()), corr_absS=float(cors[0][1]))
    A[3].axhline(1, color="k", lw=0.8, ls=":")
    lab = [r[0] for r in pk_rows]; B[1].bar(lab, [r[1] for r in pk_rows], color=[r[3] for r in pk_rows]); B[2].bar(lab, [r[2] for r in pk_rows], color=[r[3] for r in pk_rows])
    B[3].bar([r[0] for r in cors], [r[1] for r in cors], color=[r[2] for r in cors])
    for b_ in B[1:]: b_.tick_params(axis="x", rotation=30)
    B[0].axhline(1, color="0.5", lw=0.6); B[0].invert_xaxis(); B[0].set_xlabel("wavelength [px]"); B[0].set_ylabel(f"SIDM{s:g}\nP(SIDM map) / P(CDM map)", fontsize=12)
    B[1].set_ylabel("peaks per map"); B[2].set_ylabel("bright peaks per map (> 0.5 dex)"); B[3].set_ylabel("corr(|Δ|, |CDM small-scale|)")
    A[0].set_ylabel(f"SIDM{s:g}  ({m.sum()} test maps)\nmean Δ [dex]", fontsize=12); A[1].set_ylabel("rms Δ [dex]"); A[2].set_ylabel("power of Δ")
    A[3].set_ylabel("PIT density (1 = calibrated)"); A[3].set_xlabel("rank of true pixel among draws"); A[4].set_yscale("log"); A[4].set_xlabel("Δ [dex]")
    for j in (0, 1): A[j].set_xscale("log"); A[j].set_xlabel("R [kpc]")
    A[2].invert_xaxis(); A[2].set_xlabel("wavelength [px]")
for j, t_ in enumerate(["mean Δ = SIDM − CDM", "per-pixel rms of Δ (one draw)", "power spectrum of Δ", "calibration (PIT histogram)", "one-point PDF of Δ"]):
    ax[0, j].set_title(t_, fontsize=12); ax[0, j].legend(fontsize=9)
for j, t_ in enumerate(["graininess: SIDM-map power / CDM-map power (log)", "substructure peaks per map", "bright peaks per map", "is the change tied to CDM substructure?"]):
    ax2[0, j].set_title(t_, fontsize=12)
ax2[0, 0].legend(fontsize=9)
fig.suptitle("BAHAMAS DM-only, BAHAMAS → BAHAMAS test split: probabilistic converters — statistics", fontsize=15)
fig2.suptitle("BAHAMAS DM-only, BAHAMAS → BAHAMAS test split: probabilistic converters — structure", fontsize=15)
fig.savefig(f"{C.HERE}/figures/dm_compare_stats.png", dpi=72); fig2.savefig(f"{C.HERE}/figures/dm_compare_structure.png", dpi=72)
json.dump(summ, open(f"{C.HERE}/runs/dm_compare_summary.json", "w"), indent=1)
# merger
row = 1951; j = np.where((te["row"] == row) & (te["sigma"] == 1.0))[0]
if len(j):
    j = j[0]; c, t = te["c"][j], te["s"][j]; cols = [("true SIDM1", t)] + [(f"{n} draw {q + 1}", X[q, j]) for n, X in M.items() for q in range(2)]
    fig, ax = plt.subplots(2, len(cols), figsize=(4.3 * len(cols), 9), constrained_layout=True); kw = dict(origin="lower", interpolation="nearest")
    lo, hi = np.percentile(c, [1, 99.9])
    for q, (t_, X) in enumerate(cols):
        ax[0, q].imshow(X, cmap="magma", vmin=lo, vmax=hi, **kw); ax[0, q].set_title(t_, fontsize=12)
        im = ax[1, q].imshow(X - c, cmap="RdBu_r", vmin=-0.4, vmax=0.4, **kw); ax[1, q].set_title(f"− CDM, rms {np.std(X - c):.3f}", fontsize=11)
    for a_ in ax.ravel(): a_.set_xticks([]); a_.set_yticks([])
    ax[0, 0].set_ylabel("log10 Σ_DM", fontsize=13); ax[1, 0].set_ylabel("Δ [dex]", fontsize=13); fig.colorbar(im, ax=ax[1], shrink=0.6, label="dex")
    fig.suptitle(f"Merger {D['match_id'][row]} (test), DM-only: true SIDM1 and two draws per converter", fontsize=15)
    fig.savefig(f"{C.HERE}/figures/dm_compare_merger.png", dpi=72)
print(json.dumps(summ, indent=1))

"""Star - dark matter cross-correlation function of the small-scale fluctuations, BAHAMAS test split.

delta = log10 map - gaussian_filter(log10 map, HP px)   (removes the host halo; keeps galaxies / subhaloes)
rho(r) = <delta_*(x) delta_DM(x + r)> / (sigma_* sigma_DM), x and x + r both within R < 46 px (920 kpc), ring-averaged,
computed with zero-padded FFTs and summed over maps before normalising.
Models: CDM (CDM stars x CDM DM), true SIDM (SIDM stars x SIDM DM), oracle (stars-flow draw x true SIDM DM; the DM it was
conditioned on), chain (stars-flow draw d x DM-flow draw d). Also the DM-DM and star-star auto-correlations for reference.
Figure figures/star_dm_xcorr.png, rows sigma: rho_*DM(r); its ratio to CDM; auto-correlations.
usage: python star_dm_xcorr.py
"""
import json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import dm_common as C, two_scale as T2
HP, NDRAW, N = 4.0, 4, C.N; P = 2 * N
W = np.zeros((P, P)); W[:N, :N] = T2.R < 46
lag = np.hypot(*np.meshgrid(np.fft.fftfreq(P) * P, np.fft.fftfreq(P) * P, indexing="ij")); LB = lag.astype(int); RMAX = 30
cntb = np.bincount(LB.ravel(), minlength=RMAX)[:RMAX]
hp = lambda L: (L - gaussian_filter(L.astype(np.float64), (0, HP, HP))) * (T2.R < 46)


def xsum(A, B):
    """sum over maps of the masked cross-correlation numerator, and of the pair counts, per lag"""
    num = np.zeros((P, P)); FW = np.fft.rfft2(W)
    for b in range(0, len(A), 128):
        a = np.zeros((min(128, len(A) - b), P, P)); c = np.zeros_like(a); a[:, :N, :N] = A[b:b + 128]; c[:, :N, :N] = B[b:b + 128]
        num += np.fft.irfft2(np.conj(np.fft.rfft2(a)) * np.fft.rfft2(c), s=(P, P)).sum(0)
    pairs = np.fft.irfft2(np.conj(FW) * FW, s=(P, P)) * len(A)
    ring = lambda q: np.bincount(LB.ravel(), q.ravel(), minlength=RMAX)[:RMAX]
    return ring(num) / np.maximum(ring(pairs), 1)


def rho(A, B):
    va, vb = (np.sum(q[:, T2.R < 46] ** 2) / (len(q) * (T2.R < 46).sum()) for q in (A, B))
    return xsum(A, B) / np.sqrt(va * vb)


S, _ = C.load(); te = S["test"]; sig = te["sigma"]; D = np.load(f"{C.HERE}/data/pairs_stars.npz", allow_pickle=True)
xs = D["x"][D["xi"][te["k"]]][:, 2:98, 2:98]; ys = D["y"][te["k"]][:, 2:98, 2:98]
dXs, dYs, dXd, dYd = hp(xs), hp(ys), hp(te["c"]), hp(te["s"])
SO = np.load(f"{C.HERE}/runs/stars_true/draws.npz")["draws"][:NDRAW].astype(np.float32)
SC = np.load(f"{C.HERE}/runs/stars_hyb/draws.npz")["draws"][:NDRAW].astype(np.float32)
DD = np.load(f"{C.HERE}/runs/dm_flow_hyb/draws.npz")["draws"][:NDRAW].astype(np.float32)
rk = (np.arange(RMAX) + 0.5) * 20
COL = {"CDM": "0.55", "true": "k", "oracle": "#2a9d8f", "chain": "#d1495b"}
LAB = {"CDM": "CDM", "true": "true SIDM", "oracle": "flow stars × true SIDM DM", "chain": "flow stars × flow DM (chain)"}
fig, ax = plt.subplots(3, 3, figsize=(20, 15), constrained_layout=True); summ = {}
for i, s in enumerate(C.SIG):
    m = sig == s; A = ax[i]; r = {}
    r["CDM"] = rho(dXs[m], dXd[m]); r["true"] = rho(dYs[m], dYd[m])
    r["oracle"] = np.mean([rho(hp(SO[d][m]), dYd[m]) for d in range(NDRAW)], 0)
    r["chain"] = np.mean([rho(hp(SC[d][m]), hp(DD[d][m])) for d in range(NDRAW)], 0)
    for nm, v in r.items():
        lw, ls = (2.6, "-") if nm in ("CDM", "true") else (1.8, "--")
        A[0].plot(rk, v, color=COL[nm], lw=lw, ls=ls, label=LAB[nm])
        if nm != "CDM": A[1].plot(rk, v / r["CDM"], color=COL[nm], lw=lw, ls=ls, label=LAB[nm])
    auto = {"DM true": rho(dYd[m], dYd[m]), "DM chain": np.mean([rho(hp(DD[d][m]), hp(DD[d][m])) for d in range(2)], 0),
            "stars true": rho(dYs[m], dYs[m]), "stars oracle": np.mean([rho(hp(SO[d][m]), hp(SO[d][m])) for d in range(2)], 0)}
    for nm, v in auto.items():
        A[2].plot(rk, v, color="#3a6fb0" if nm.startswith("DM") else "#e9a03b", lw=2.4 if "true" in nm else 1.6, ls="-" if "true" in nm else "--", label=nm)
    summ[str(s)] = {nm: dict(rho0=float(v[0]), rho40=float(v[2]), rho100=float(v[5]), zero_cross_kpc=float(rk[np.argmax(v < 0)]) if (v < 0).any() else None) for nm, v in r.items()}
    A[0].axhline(0, color="0.6", lw=0.6); A[1].axhline(1, color="0.6", lw=0.6); A[1].set_ylim(0.5, 1.3); A[2].axhline(0, color="0.6", lw=0.6)
    A[0].set_ylabel(f"SIDM{s:g} ({m.sum()} test maps)\nρ_*DM(r)", fontsize=12); A[1].set_ylabel("ρ_*DM / ρ_*DM,CDM"); A[2].set_ylabel("auto-correlation ρ(r)")
    for a in A: a.set_xlabel("separation r [kpc]"); a.legend(fontsize=9); a.set_xlim(0, 400)
    A[1].set_xlim(0, 110); A[1].set_ylim(0.85, 1.25)
for j, t in enumerate(["star – DM cross-correlation (high-pass, 80 kpc)", "relative to CDM (r < 110 kpc, where ρ > 0.05)", "auto-correlations (DM blue, stars orange)"]):
    ax[0, j].set_title(t, fontsize=13)
fig.suptitle(f"Star – dark matter cross-correlation of small-scale structure (log Σ minus {HP:g} px Gaussian), BAHAMAS test split", fontsize=15)
fig.savefig(f"{C.HERE}/figures/star_dm_xcorr.png", dpi=65); json.dump(summ, open(f"{C.HERE}/runs/star_dm_xcorr_summary.json", "w"), indent=1)
for s, d in summ.items():
    for nm, v in d.items(): print(f"SIDM{s} {nm:7s} rho(0)={v['rho0']:.3f} rho(40kpc)={v['rho40']:.3f} rho(100kpc)={v['rho100']:.3f} zero-crossing {v['zero_cross_kpc']} kpc")
print("wrote figures/star_dm_xcorr.png")

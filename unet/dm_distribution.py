"""Probabilistic CDM -> SIDM converter on BAHAMAS dark-matter-only maps (BAHAMAS -> BAHAMAS: fit on the train split,
scored on the test split). data/pairs_dm.npz, |shift| <= 2 px, central 96x96 px.

Mean   : the two-scale converter (two_scale.py, no CNN): radial template A(r) sigma^p + rank-based small-scale map.
Spread : residual e = SIDM - mean is modelled as e = a(x) g,
         a(x) = rms of training residuals in bins of (radius, CDM small-scale contrast S_c = c - G(c, 2 px)) per sigma
                (subhalo position scatter between runs lives where the CDM map has sharp structure; particle noise
                rises outwards),
         g    = unit-variance stationary Gaussian field with the azimuthal power spectrum of e/a in training.
Sample : SIDM = mean + a(x) g, any number of draws per CDM map.
Checks on test clusters: mean residual, rms(r) of SIDM - CDM (true vs sampled vs mean-only), power spectrum of
SIDM - CDM, coverage of the 68/95 % intervals vs radius, one-point PDF; merger (row 1951) true vs samples.
usage: python3 dm_distribution.py -> runs/dm_distribution/{model.npz,summary.json}, figures/dm_distribution_{summary,merger}.png
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import two_scale as T2
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/runs/dm_distribution"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1); N = 96; SIG = (0.1, 0.3, 1.0)
D = np.load(f"{HERE}/data/pairs_dm.npz", allow_pickle=True)
x, y, xi, sig, sh, spl, zz, ok = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift", "ok"))
sel = ok & (np.abs(sh).max(1) <= 2) & (sig > 0)
def part(lab):
    k = np.where(sel & (spl[xi] == lab))[0]
    return dict(k=k, c=x[xi[k], 2:98, 2:98].astype(np.float32), s=y[k, 2:98, 2:98].astype(np.float32), sigma=sig[k].astype(np.float32),
                z=zz[xi[k]].astype(np.float32), row=xi[k])
tr, te = part("train"), part("test")
print("train", len(tr["k"]), "test", len(te["k"]), flush=True)
tpl = T2.fit_template(tr); sm = T2.fit_small(tr)
print(f"DM template: core {tpl['A'][0]:+.4f} sigma^{tpl['p']:.3f}; small p2 {sm['p2']:.3f}", flush=True)
def mean_pred(P):
    out = np.empty_like(P["c"])
    for b in range(0, len(out), 300):
        c = P["c"][b:b + 300].astype(np.float64); Lc = T2.smooth(c); Sc = c - Lc; s_ = P["sigma"][b:b + 300].astype(np.float64)
        out[b:b + 300] = c + T2.predict_large(Lc, s_, P["z"][b:b + 300].astype(np.float64), tpl) + T2.apply_small(Sc, s_, sm) - Sc
    return out
for P in (tr, te): P["mu"] = mean_pred(P); P["Sc"] = P["c"] - T2.smooth(P["c"].astype(np.float64)).astype(np.float32)
# amplitude table a[sigma, rbin, sbin]
REDG = np.array([0, 2, 4, 7, 11, 16, 24, 34, 49]); RBIN = np.minimum(np.digitize(T2.R, REDG) - 1, len(REDG) - 2)
SEDG = np.quantile(tr["Sc"][:, T2.R > 4], np.linspace(0, 1, 9)[1:-1]); NS = len(SEDG) + 1
amp = np.zeros((len(SIG), len(REDG) - 1, NS))
e_tr = tr["s"] - tr["mu"]
for i, s in enumerate(SIG):
    m = tr["sigma"] == s; e = e_tr[m]; sb = np.digitize(tr["Sc"][m], SEDG); rb = np.broadcast_to(RBIN, e.shape)
    for j in range(len(REDG) - 1):
        for q in range(NS):
            w = (rb == j) & (sb == q); amp[i, j, q] = np.sqrt(np.mean(e[w] ** 2)) if w.sum() > 50 else np.nan
    # fill sparse bins from neighbours in S_c
    for j in range(len(REDG) - 1):
        v = amp[i, j]; f = np.isfinite(v); amp[i, j] = np.interp(np.arange(NS), np.where(f)[0], v[f])
def a_map(P):
    si = np.searchsorted(SIG, P["sigma"]); sb = np.digitize(P["Sc"], SEDG)
    return amp[si[:, None, None], RBIN[None], sb].astype(np.float32)
for P in (tr, te): P["a"] = a_map(P)
# a(x) smoothed 2 px (bin edges would add small-scale power), then: marginal of g = e / a from training quantiles
# (heavy-tailed, not Gaussian), and a stationary filter tuned so the power of a*g matches the training residual power
from scipy.ndimage import gaussian_filter as _G
from scipy.special import ndtr
for P in (tr, te): P["a"] = _G(P["a"], (0, 2, 2), mode="nearest").astype(np.float32)
kk = np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")) * N; KB = np.minimum(kk.astype(int), N // 2)
QL = np.linspace(0.0005, 0.9995, 400); GQ = np.quantile((e_tr / tr["a"])[::3], QL)
def praw(M):
    M = M - M.mean((1, 2), keepdims=True); p = (np.abs(np.fft.fft2(M)) ** 2).mean(0)
    return np.bincount(KB.ravel(), p.ravel()) / np.bincount(KB.ravel())
sub = {k: (v[::10] if isinstance(v, np.ndarray) and len(v) == len(tr["c"]) else v) for k, v in tr.items()}
g0 = (e_tr / tr["a"])[::5]; FILT = np.sqrt(praw(g0)[KB]); FILT[0, 0] = 0
def field(n, m, filt):
    w = rng.standard_normal((n, m, N, N)).astype(np.float32)
    f = np.fft.ifft2(np.fft.fft2(w) * filt).real; f /= f.std((2, 3), keepdims=True)
    return np.interp(ndtr(f), QL, GQ).astype(np.float32)  # Gaussian copula -> empirical marginal
def draw(P, n=1, filt=None):
    return P["mu"][None] + P["a"][None] * field(n, len(P["c"]), FILT if filt is None else filt)
target = praw(e_tr[::10])
for it in range(4):
    ratio = target / np.maximum(praw((draw(sub, 1)[0] - sub["mu"])), 1e-12); ratio[0] = 1
    FILT = FILT * np.sqrt(ratio)[KB]; FILT[0, 0] = 0
    print(f"spectral iteration {it}: max |log10 ratio| {np.abs(np.log10(ratio[1:N // 2])).max():.3f}", flush=True)
np.savez(f"{OUT}/model.npz", amp=amp, redg=REDG, sedg=SEDG, filt=FILT, sigmas=SIG, ql=QL, gq=GQ)
json.dump(dict(template=tpl, small=sm), open(f"{OUT}/template.json", "w"))
# ---------- evaluation on test
smp = draw(te, 1)[0]
cnt = np.bincount(T2.RB.ravel()); Rk = np.arange(48) * 20 + 10
prof = lambda M: np.bincount(T2.RB.ravel(), M.ravel(), 48) / cnt
def pspec(M):
    M = M - M.mean((1, 2), keepdims=True); w = np.hanning(N)[:, None] * np.hanning(N)[None]
    p = (np.abs(np.fft.fft2(M * w)) ** 2).mean(0); return (np.bincount(KB.ravel(), p.ravel()) / np.bincount(KB.ravel()))[1:N // 2]
summ = {}
fig, ax = plt.subplots(3, 5, figsize=(26, 14.5), constrained_layout=True)
CL = dict(true="k", sample="#c0392b", mean="#3a6fb0")
for i, s in enumerate(SIG):
    m = te["sigma"] == s; c, t, mu, sp, a = te["c"][m], te["s"][m], te["mu"][m], smp[m], te["a"][m]
    dT, dS, dM = t - c, sp - c, mu - c; A = ax[i]
    # 1 mean Delta profile
    for d_, lab in ((dT, "true"), (dM, "mean"), (dS, "sample")):
        A[0].plot(Rk, prof(d_.mean(0)), color=CL[lab], lw=2 if lab == "true" else 1.6, ls="--" if lab == "sample" else "-", label=lab)
    A[0].axhline(0, color="0.5", lw=0.6); A[0].set_xscale("log"); A[0].set_ylabel(f"SIDM{s:g}  ({m.sum()} test maps)\nmean Δ = SIDM − CDM [dex]", fontsize=12)
    # 2 rms of Delta
    for d_, lab in ((dT, "true"), (dM, "mean"), (dS, "sample")):
        A[1].plot(Rk, np.sqrt(prof((d_ ** 2).mean(0))), color=CL[lab], lw=2, ls="--" if lab == "sample" else "-", label=lab)
    A[1].set_xscale("log"); A[1].set_ylim(0, None); A[1].set_ylabel("rms Δ per pixel [dex]")
    # 3 power of Delta
    kx = np.arange(1, N // 2)
    for d_, lab in ((dT, "true"), (dM, "mean"), (dS, "sample")):
        A[2].loglog(N / kx, pspec(d_), color=CL[lab], lw=2, ls="--" if lab == "sample" else "-", label=lab)
    A[2].invert_xaxis(); A[2].set_xlabel("wavelength [px]"); A[2].set_ylabel("power of Δ")
    # 4 coverage
    gz = (t - mu) / a; lo68, hi68, lo95, hi95 = np.interp([0.16, 0.84, 0.025, 0.975], QL, GQ)
    in68 = (gz > lo68) & (gz < hi68); in95 = (gz > lo95) & (gz < hi95); c68 = prof(in68.mean(0)); c95 = prof(in95.mean(0))
    A[3].plot(Rk, c68, color="#2a9d8f", lw=2, label="inside 68 % interval"); A[3].plot(Rk, c95, color="#e9a03b", lw=2, label="inside 95 % interval")
    A[3].axhline(0.68, color="#2a9d8f", ls=":"); A[3].axhline(0.95, color="#e9a03b", ls=":"); A[3].set_xscale("log"); A[3].set_ylim(0.4, 1.0)
    A[3].set_ylabel("fraction of true pixels")
    # 5 one-point of Delta
    bins = np.linspace(-0.8, 0.8, 81)
    for d_, lab in ((dT, "true"), (dM, "mean"), (dS, "sample")):
        A[4].hist(d_[:, T2.FIELD].ravel(), bins, histtype="step", density=True, color=CL[lab], lw=2, ls="--" if lab == "sample" else "-", label=lab)
    A[4].set_yscale("log"); A[4].set_xlabel("Δ [dex]"); A[4].set_ylabel("pixel PDF")
    for j in (0, 1, 3): A[j].set_xlabel("R [kpc]")
    summ[str(s)] = dict(n=int(m.sum()), rms_true=float(dT.std()), rms_sample=float(dS.std()), rms_mean=float(dM.std()),
                        cov68=float(in68.mean()), cov95=float(in95.mean()),
                        core_true=float(dT[:, T2.CORE].mean()), core_mean=float(dM[:, T2.CORE].mean()))
    print(s, summ[str(s)], flush=True)
for j, t_ in enumerate(["mean Δ profile (test average)", "per-pixel rms of Δ", "power spectrum of Δ", "calibration: coverage vs radius", "one-point PDF of Δ (r < 46 px)"]):
    ax[0, j].set_title(t_, fontsize=12); ax[0, j].legend(fontsize=9)
fig.suptitle("BAHAMAS dark-matter-only maps, BAHAMAS → BAHAMAS: probabilistic converter (mean + spatially varying, heavy-tailed, spectrum-matched noise) on held-out test clusters", fontsize=15)
fig.savefig(f"{HERE}/figures/dm_distribution_summary.png", dpi=75)
json.dump(dict(test=summ, template=tpl), open(f"{OUT}/summary.json", "w"), indent=1)
# ---------- merger
row = 1951; mk = np.where(te["row"] == row)[0]
if len(mk):
    mm = {float(te["sigma"][j]): j for j in mk}; s = 1.0 if 1.0 in mm else max(mm); j = mm[s]
    P1 = {k: (v[j:j + 1] if isinstance(v, np.ndarray) and len(v) == len(te["c"]) else v) for k, v in te.items()}
    S3 = draw(P1, 3)[:, 0]; c, t, mu = te["c"][j], te["s"][j], te["mu"][j]
    fig, ax = plt.subplots(2, 5, figsize=(26, 10.5), constrained_layout=True); kw = dict(origin="lower", interpolation="nearest")
    lo, hi = np.percentile(c, [1, 99.9])
    for q, (M, t_) in enumerate([(t, f"true SIDM{s:g}"), (mu, "mean prediction"), (S3[0], "sample 1"), (S3[1], "sample 2"), (S3[2], "sample 3")]):
        ax[0, q].imshow(M, cmap="magma", vmin=lo, vmax=hi, **kw); ax[0, q].set_title(t_, fontsize=13)
        im = ax[1, q].imshow(M - c, cmap="RdBu_r", vmin=-0.4, vmax=0.4, **kw); ax[1, q].set_title(f"{t_} − CDM, rms {np.std(M - c):.3f}", fontsize=12)
    for a_ in ax.ravel(): a_.set_xticks([]); a_.set_yticks([])
    ax[0, 0].set_ylabel("log10 Σ_DM", fontsize=13); ax[1, 0].set_ylabel("Δ [dex]", fontsize=13); fig.colorbar(im, ax=ax[1], shrink=0.7, label="dex")
    fig.suptitle(f"Merger {D['match_id'][row]} (test cluster), dark-matter-only: true SIDM{s:g} vs mean and three draws from the probabilistic converter", fontsize=15)
    fig.savefig(f"{HERE}/figures/dm_distribution_merger.png", dpi=75)
print("done")

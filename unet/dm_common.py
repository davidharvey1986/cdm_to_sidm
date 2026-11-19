"""Shared pieces for the probabilistic CDM -> SIDM converters on BAHAMAS dark-matter-only maps (data/pairs_dm.npz),
BAHAMAS -> BAHAMAS: fit on the train split, test split held out. Central 96x96 px, |shift| <= 2 px, sigma > 0.
mean model = two-scale converter (template + small-scale rank map) refitted on DM maps (runs/dm_distribution/template.json).
"""
import os, json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr
import two_scale as T2
HERE = os.path.dirname(os.path.abspath(__file__)); N = 96; SIG = (0.1, 0.3, 1.0)
REDG = np.array([0, 2, 4, 7, 11, 16, 24, 34, 49]); RBIN = np.minimum(np.digitize(T2.R, REDG) - 1, len(REDG) - 2)
KK = np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")) * N; KB = np.minimum(KK.astype(int), N // 2)
TPL = json.load(open(f"{HERE}/runs/dm_distribution/template.json"))


def load():
    D = np.load(f"{HERE}/data/pairs_dm.npz", allow_pickle=True)
    x, y, xi, sig, sh, spl, zz, ok = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift", "ok"))
    sel = ok & (np.abs(sh).max(1) <= 2) & (sig > 0); out = {}
    for lab in ("train", "val", "test"):
        k = np.where(sel & (spl[xi] == lab))[0]
        P = dict(k=k, row=xi[k], c=x[xi[k], 2:98, 2:98].astype(np.float32), s=y[k, 2:98, 2:98].astype(np.float32),
                 sigma=sig[k].astype(np.float32), z=zz[xi[k]].astype(np.float32))
        P["Sc"] = (P["c"] - T2.smooth(P["c"].astype(np.float64))).astype(np.float32)
        P["Lc"], P["dL"], P["Sm"] = mean_parts(P); P["mu"] = P["Lc"] + P["dL"] + P["Sm"]
        out[lab] = P
    return out, D


def mean_parts(P):
    """large-scale CDM, template change, mapped small-scale part (mean = Lc + dL + Sm)"""
    Lc = np.empty_like(P["c"]); dL = np.empty_like(P["c"]); Sm = np.empty_like(P["c"])
    for b in range(0, len(Lc), 300):
        c = P["c"][b:b + 300].astype(np.float64); L = T2.smooth(c); S = c - L; s_ = P["sigma"][b:b + 300].astype(np.float64)
        Lc[b:b + 300] = L; dL[b:b + 300] = T2.predict_large(L, s_, P["z"][b:b + 300].astype(np.float64), TPL["template"])
        Sm[b:b + 300] = T2.apply_small(S, s_, TPL["small"])
    return Lc, dL, Sm


def sub(P, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(P["c"]) else v) for k, v in P.items()}


def sedges(P):
    return np.quantile(P["Sc"][:, T2.R > 4], np.linspace(0, 1, 9)[1:-1])


def amp_table(e, P, SEDG):
    """rms of e in (sigma, radial bin, CDM small-scale contrast bin)"""
    NS = len(SEDG) + 1; amp = np.zeros((len(SIG), len(REDG) - 1, NS))
    for i, s in enumerate(SIG):
        m = P["sigma"] == s; ee = e[m] ** 2; sb = np.digitize(P["Sc"][m], SEDG); rb = np.broadcast_to(RBIN, ee.shape)
        idx = (rb * NS + sb).ravel(); n = np.bincount(idx, minlength=amp[0].size); v = np.bincount(idx, ee.ravel(), amp[0].size)
        a = np.where(n > 50, v / np.maximum(n, 1), np.nan).reshape(len(REDG) - 1, NS)
        for j in range(len(REDG) - 1):
            f = np.isfinite(a[j]); a[j] = np.interp(np.arange(NS), np.where(f)[0], a[j][f]) if f.any() else 0
        amp[i] = np.sqrt(a)
    return amp


def a_map(amp, P, SEDG):
    si = np.searchsorted(SIG, P["sigma"]); sb = np.digitize(P["Sc"], SEDG)
    return gaussian_filter(amp[si[:, None, None], RBIN[None], sb], (0, 2, 2), mode="nearest").astype(np.float32)


def praw(M):
    M = M - M.mean((-2, -1), keepdims=True); p = (np.abs(np.fft.fft2(M)) ** 2).reshape(-1, N, N).mean(0)
    return np.bincount(KB.ravel(), p.ravel()) / np.bincount(KB.ravel())


def gfield(rng, shape, filt, marg=None):
    """stationary unit-variance field with filter filt; optional Gaussian-copula marginal (QL, GQ)"""
    w = rng.standard_normal(shape).astype(np.float32)
    f = np.fft.ifft2(np.fft.fft2(w) * filt).real; f /= f.std((-2, -1), keepdims=True)
    return f.astype(np.float32) if marg is None else np.interp(ndtr(f), *marg).astype(np.float32)


def peaks(m):
    """per-map peak count (high-pass > 0.15 dex, local max in 5x5, r > 6 px), bright (> 0.5 dex)"""
    from scipy.ndimage import maximum_filter
    hp = m - gaussian_filter(m, (0, 3, 3)); pk = (maximum_filter(hp, (1, 5, 5)) == hp) & (hp > 0.15) & (T2.R > 6)[None]
    pk[:, :6] = pk[:, -6:] = 0; pk[:, :, :6] = pk[:, :, -6:] = 0
    return pk.sum((1, 2)), (pk & (hp > 0.5)).sum((1, 2))


MC_AP = T2.R < 45  # mass-conservation aperture: R < 900 kpc
# true log10 M_SIDM/M_CDM within 900 kpc on the TRAIN split (mean, sd) per sigma: SIDM moves mass out of the core but keeps the total
MC_TRAIN = {0.1: (-0.0008, 0.0098), 0.3: (-0.0002, 0.0096), 1.0: (0.0018, 0.0102)}


def conserve(X, c, sigma=None, seed=0, ap=MC_AP):
    """rescale each draw (a constant shift in log Sigma) so its mass inside ap equals the CDM map's mass times a ratio drawn
    from the train-split SIDM/CDM distribution MC_TRAIN[sigma] (sigma=None: exactly the CDM mass).
    X (..., n, 96, 96), c (n, 96, 96) log10 Sigma; sigma (n,). The flow does not conserve mass per map (draw scatter 0.016 dex
    vs true 0.010); forcing exactly the CDM mass removes the true scatter and makes the pixel calibration under-dispersed."""
    mX = np.log10((10 ** X[..., ap].astype(np.float64)).sum(-1)); mc = np.log10((10 ** c[:, ap].astype(np.float64)).sum(-1))
    tgt = mc
    if sigma is not None:
        mu = np.array([MC_TRAIN[round(float(q), 3)][0] for q in sigma]); sd = np.array([MC_TRAIN[round(float(q), 3)][1] for q in sigma])
        tgt = mc + mu + sd * np.random.default_rng(seed).standard_normal(mX.shape)
    return (X + (tgt - mX)[..., None, None]).astype(X.dtype)

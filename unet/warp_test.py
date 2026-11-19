"""Warp test: split SIDM - CDM into a smooth displacement field u and a density change delta.

  SIDM(x + u(x)) * J(x) = CDM(x) * 10^delta(x),   J = det(I + grad u)  (mass-conserving pull-back; J clipped to [JMIN, 1/JMIN] where the warp folds)

u is estimated from rigid shifts in overlapping windows (WIN px, stride STRIDE): normalised cross-correlation
of the high-passed log maps (structure on < HP px scales: subhaloes), parabolic sub-pixel peak, |u| <= MAXU.
Windows with a weak correlation peak (< CMIN) get no vote; votes are weighted by peak x structure energy, smoothed and
interpolated to every pixel. The central halo is aligned already, and a radial expansion there IS the SIDM core,
so u is tapered to 0 inside R0 px of the centre (linear ramp to R1).
Diagnostics: aperture mass change at the CDM subcluster peak and at the centre, before vs after the warp;
rms of delta; |u| vs sigma. Output runs/warp_test/{summary.json, example_<row>.png}, data/warp_u.npz
usage: python3 warp_test.py [--example 1951]
"""
import os, json, argparse
import numpy as np, matplotlib
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--example", type=int, default=1951); a = ap.parse_args()
WIN, STRIDE, HP, MAXU, CMIN, R0, R1, SM, JMIN = 16, 4, 4.0, 6, 0.5, 5.0, 10.0, 0.6, 0.25
OUT = f"{HERE}/runs/warp_test"; os.makedirs(OUT, exist_ok=True)
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig, sh = (D[k] for k in ("x", "y", "xi", "sigma", "shift"))
pk = np.load(f"{HERE}/data/bahamas_disturbance.npz")["second_peak_ratio"]
ev = np.where((np.abs(sh).max(1) <= 2) & (sig > 0))[0]
n = 100; yy, xx = np.indices((n, n)); r = np.hypot(yy - 49.5, xx - 49.5)
taper = np.clip((r - R0) / (R1 - R0), 0, 1)
hann = np.outer(np.hanning(WIN), np.hanning(WIN))
gc = np.arange(WIN // 2, n - WIN // 2 + 1, STRIDE) - 0.5  # window centres (px coords)
d = (np.arange(WIN) + WIN // 2) % WIN - WIN // 2


def shifts(c, t):
    """per-window (dy, dx) such that t(x + u) ~ c(x), and the peak correlation"""
    hc, ht = c - gaussian_filter(c, HP), t - gaussian_filter(t, HP)
    pc = swv(hc, (WIN, WIN))[::STRIDE, ::STRIDE] * hann; pt = swv(ht, (WIN, WIN))[::STRIDE, ::STRIDE] * hann
    pc = pc - pc.mean((-2, -1), keepdims=True); pt = pt - pt.mean((-2, -1), keepdims=True)
    xc = np.fft.ifft2(np.conj(np.fft.fft2(pc)) * np.fft.fft2(pt)).real
    en = np.sqrt((pc ** 2).sum((-2, -1)) * (pt ** 2).sum((-2, -1)))
    xc /= en[..., None, None] + 1e-12
    ok = (np.abs(d)[:, None] <= MAXU) & (np.abs(d)[None, :] <= MAXU)
    xc = np.where(ok, xc, -np.inf)
    G = xc.shape[0]; flat = xc.reshape(G, G, -1); im = flat.argmax(-1); iy, ix = im // WIN, im % WIN
    peak = np.take_along_axis(flat, im[..., None], -1)[..., 0]
    def sub(i0, axis):
        g = lambda o: np.take_along_axis(flat, (((iy + o) % WIN) * WIN + ix if axis == 0 else iy * WIN + (ix + o) % WIN)[..., None], -1)[..., 0]
        m1, p0, p1 = g(-1), peak, g(1); den = m1 - 2 * p0 + p1
        return d[i0] + np.where(np.isfinite(den) & (den < 0), 0.5 * (m1 - p1) / np.where(den < 0, den, -1), 0)
    return sub(iy, 0), sub(ix, 1), peak, en


def field(c, t):
    uy, ux, pkc, en = shifts(c, t)
    w = (pkc >= CMIN) * pkc * en / (np.median(en) + 1e-12)  # windows with strong, well-matched structure dominate
    sm = lambda v: gaussian_filter(v * w, SM) / (gaussian_filter(w, SM) + 1e-6)
    uy, ux = sm(uy), sm(ux)
    gi = np.interp(np.arange(n), gc, np.arange(len(gc)))
    Y, X = np.meshgrid(gi, gi, indexing="ij")
    Uy = map_coordinates(uy, [Y, X], order=1, mode="nearest") * taper
    Ux = map_coordinates(ux, [Y, X], order=1, mode="nearest") * taper
    return Uy, Ux


def warp(t_log, Uy, Ux):
    T = 10.0 ** t_log.astype(np.float64)
    W = map_coordinates(T, [yy + Uy, xx + Ux], order=1, mode="nearest")
    dyy, dyx = np.gradient(Uy); dxy, dxx = np.gradient(Ux)
    J = (1 + dyy) * (1 + dxx) - dyx * dxy
    return np.log10(W * np.clip(J, JMIN, 1 / JMIN)), J


nfold = nJ = 0
U = np.zeros((len(ev), 2, n, n), np.float32); YW = np.empty((len(ev), n, n), np.float32)
for k, i in enumerate(ev):
    c, t = x[xi[i]].astype(np.float64), y[i].astype(np.float64)
    Uy, Ux = field(c, t); U[k] = Uy, Ux; YW[k], J = warp(t, Uy, Ux); nfold += int((J < JMIN).sum()); nJ += J.size
    if k % 2000 == 0:
        print(f"  {k}/{len(ev)}", flush=True)
print(f"Jacobian clipped (J < {JMIN}, local fold-over) in {nfold / nJ:.4%} of pixels")
np.savez_compressed(f"{HERE}/data/warp_u.npz", pair=ev, u=U.astype(np.float16), y_warped=YW)

# ---- diagnostics
res = {}
AP = (1, 2, 4, 8)
sel = pk[xi[ev]] >= 0.5
for s in (0.1, 0.3, 1.0):
    R = {}
    for tag, Ymap in (("before", y[ev]), ("after", YW)):
        sub = {q: [] for q in AP}; cen = {q: [] for q in AP}
        for k in np.where(sel & (sig[ev] == s))[0]:
            i = ev[k]; c = 10.0 ** x[xi[i]].astype(float); t = 10.0 ** Ymap[k].astype(float)
            L = gaussian_filter(c, 2); mx = (maximum_filter(L, 9) == L) & (r > 8) & (r < 42)
            if not mx.any(): continue
            iy, ix = np.unravel_index(np.argmax(np.where(mx, L, 0)), L.shape); dd = np.hypot(yy - iy, xx - ix)
            for q in AP:
                sub[q].append(np.log10(t[dd < q].sum() / c[dd < q].sum())); cen[q].append(np.log10(t[r < q].sum() / c[r < q].sum()))
        R[tag] = dict(subcluster={q: float(np.median(v)) for q, v in sub.items()}, centre={q: float(np.median(v)) for q, v in cen.items()},
                      subcluster_r1_std=float(np.std(sub[1])))
    m = sig[ev] == s; dl0 = (y[ev[m]] - x[xi[ev[m]]])[:, 5:95, 5:95]; dl1 = (YW[m] - x[xi[ev[m]]])[:, 5:95, 5:95]
    sm4 = lambda v: gaussian_filter(v, (0, 4, 4))
    R["rms_delta"] = dict(before=float(dl0.std()), after=float(dl1.std()))
    R["rms_delta_smoothed4"] = dict(before=float(sm4(dl0).std()), after=float(sm4(dl1).std()))
    um = np.hypot(U[m, 0], U[m, 1])[:, 5:95, 5:95]
    R["u_median_px"] = float(np.median(um)); R["u_p90_px"] = float(np.percentile(um, 90))
    R["u_mean_divergence"] = float(np.mean([np.gradient(U[k, 0], axis=0) + np.gradient(U[k, 1], axis=1) for k in np.where(m)[0][:300]]))
    res[str(s)] = R
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
for s in ("0.1", "0.3", "1.0"):
    R = res[s]
    f = lambda dct: "  ".join(f"r<{q}: {v:+.3f}" for q, v in dct.items())
    print(f"sigma {s}:  |u| median {R['u_median_px']:.2f} px, 90% {R['u_p90_px']:.2f} px; mean div u {R['u_mean_divergence']:+.4f}")
    for tag in ("before", "after"):
        print(f"   {tag:6s} subcluster {f(R[tag]['subcluster'])}  (r<1 scatter {R[tag]['subcluster_r1_std']:.3f})")
        print(f"   {tag:6s} centre     {f(R[tag]['centre'])}")
    print(f"   rms delta {R['rms_delta']['before']:.3f} -> {R['rms_delta']['after']:.3f};  4px-smoothed {R['rms_delta_smoothed4']['before']:.3f} -> {R['rms_delta_smoothed4']['after']:.3f}")

# ---- example
for s in (1.0,):
    k = np.where((xi[ev] == a.example) & (sig[ev] == s))[0]
    if not len(k): break
    k = k[0]; i = ev[k]
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.8), constrained_layout=True)
    lo, hi = np.percentile(x[xi[i]], [1, 99.9])
    axs[0].imshow(x[xi[i]], cmap="magma", vmin=lo, vmax=hi, origin="lower"); axs[0].set_title("CDM")
    axs[1].imshow(y[i], cmap="magma", vmin=lo, vmax=hi, origin="lower"); axs[1].set_title(f"SIDM σ={s}")
    axs[2].imshow(y[i] - x[xi[i]], cmap="RdBu_r", vmin=-.3, vmax=.3, origin="lower"); axs[2].set_title("Δ before warp")
    q = 4; axs[3].imshow(x[xi[i]], cmap="Greys", origin="lower", alpha=0.5)
    axs[3].quiver(xx[::q, ::q], yy[::q, ::q], U[k, 1][::q, ::q], U[k, 0][::q, ::q], np.hypot(U[k, 0], U[k, 1])[::q, ::q],
                  angles="xy", scale_units="xy", scale=0.5, cmap="viridis")
    axs[3].set_title("displacement u (arrows ×2), CDM → SIDM")
    im = axs[4].imshow(YW[k] - x[xi[i]], cmap="RdBu_r", vmin=-.3, vmax=.3, origin="lower"); axs[4].set_title("δ after warp (density change)")
    fig.colorbar(im, ax=axs[[2, 4]], shrink=0.8, label="dex")
    for g in axs: g.set_xticks([]); g.set_yticks([])
    fig.suptitle(f"warp test, BAHAMAS row {a.example}, σ={s}")
    fig.savefig(f"{OUT}/example_{a.example}.png", dpi=100)
print("wrote", OUT)

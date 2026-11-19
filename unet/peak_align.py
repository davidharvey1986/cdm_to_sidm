"""Per-subhalo alignment of SIDM maps onto their CDM partner, to strip run-to-run position noise.

For each pair (CDM c, SIDM s, log10 maps, s already re-centred on the main halo):
 1. subhalo peaks in CDM: local maxima (NBR px) of the high-passed log map hp = c - G(c, HP) with hp > THR dex,
    outside the main halo (r > RMIN) and EDGE px from the border.
 2. match: normalised cross-correlation of PATCH x PATCH Hann-windowed patches of hp_c and hp_s around the
    CDM peak, shifts |u| <= MAXU, parabolic sub-pixel peak; accepted if NCC >= CMIN.
 3. move: the subhalo is its excess over the host in linear Sigma, E = max(S - host, 0), the host interpolated
    under it by normalised convolution of the map outside the cut-out; cut out with a
    taper w (1 inside AP px of the SIDM peak, 0 beyond AP + TAP). The blob B = E w is removed and pasted back
    shifted by -u (bilinear), so the smooth host is untouched and the subhalo's mass is conserved.
    Peaks are processed brightest first on the running map.
delta_aligned = log10 aligned SIDM - log10 CDM.
Outputs data/peak_align.npz (pair rows, aligned maps, peak table), runs/peak_align/{summary.json, example_<row>.png}
usage: python3 peak_align.py [--example 1951] [--selftest]
"""
import os, json, argparse
import numpy as np, matplotlib
from scipy.ndimage import gaussian_filter, maximum_filter, shift as nd_shift
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HP, THR, NBR, RMIN, EDGE, PATCH, MAXU, CMIN, AP, TAP = 3.0, 0.15, 5, 6.0, 6, 13, 6, 0.5, 2.5, 2.5
n = 100; yy, xx = np.indices((n, n)); r = np.hypot(yy - 49.5, xx - 49.5)
hw = PATCH // 2; hann = np.outer(np.hanning(PATCH + 2)[1:-1], np.hanning(PATCH + 2)[1:-1])


def peaks(c):
    hp = c - gaussian_filter(c, HP)
    m = (maximum_filter(hp, NBR) == hp) & (hp > THR) & (r > RMIN)
    m[:EDGE] = m[-EDGE:] = 0; m[:, :EDGE] = m[:, -EDGE:] = 0
    iy, ix = np.nonzero(m); o = np.argsort(-hp[iy, ix])
    return iy[o], ix[o], hp[iy[o], ix[o]], hp


def match(hpc, hps, iy, ix):
    """shift u = (dy, dx) such that the SIDM subhalo sits at (iy, ix) + u"""
    pad = hw + MAXU + 1
    C = np.pad(hpc, pad, mode="reflect"); S = np.pad(hps, pad, mode="reflect")
    yc, xc = iy + pad, ix + pad
    a = C[yc - hw:yc + hw + 1, xc - hw:xc + hw + 1] * hann; a = a - a.mean()
    best = np.full((2 * MAXU + 1, 2 * MAXU + 1), -np.inf)
    for i, dy in enumerate(range(-MAXU, MAXU + 1)):
        for j, dx in enumerate(range(-MAXU, MAXU + 1)):
            if dy * dy + dx * dx > MAXU * MAXU: continue
            b = S[yc + dy - hw:yc + dy + hw + 1, xc + dx - hw:xc + dx + hw + 1] * hann; b = b - b.mean()
            best[i, j] = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12)
    i, j = np.unravel_index(np.argmax(best), best.shape); pk = best[i, j]
    def sub(v, k, L):
        if 0 < k < L - 1 and np.isfinite(v[k - 1]) and np.isfinite(v[k + 1]):
            den = v[k - 1] - 2 * v[k] + v[k + 1]
            return 0.5 * (v[k - 1] - v[k + 1]) / den if den < 0 else 0.0
        return 0.0
    dy = i - MAXU + sub(best[:, j], i, best.shape[0]); dx = j - MAXU + sub(best[i, :], j, best.shape[1])
    return dy, dx, pk


def align(c, s):
    iy, ix, amp, hpc = peaks(c); hps = s - gaussian_filter(s, HP)
    S = 10.0 ** s.astype(np.float64); tab = []
    for y0, x0, a0 in zip(iy, ix, amp):
        dy, dx, pk = match(hpc, hps, y0, x0)
        tab.append((y0, x0, a0, dy, dx, pk))
        if pk < CMIN or (dy == 0 and dx == 0): continue
        d = np.hypot(yy - (y0 + dy), xx - (x0 + dx)); w = np.clip((AP + TAP - d) / TAP, 0, 1)
        out = (d > AP + TAP).astype(float)  # host under the subhalo: smooth interpolation from the surrounding ring
        E = np.maximum(S - gaussian_filter(S * out, HP) / np.maximum(gaussian_filter(out, HP), 1e-3), 0)
        B = E * w
        S = S - B + nd_shift(B, (-dy, -dx), order=1, mode="constant")
        S = np.maximum(S, 1e-3 * gaussian_filter(S, HP))
        hps = np.log10(S) - gaussian_filter(np.log10(S), HP)
    return np.log10(S).astype(np.float32), np.array(tab, dtype=float).reshape(-1, 6)


def selftest():
    host = lambda: 1e13 + 1e15 * np.exp(-r ** 2 / (2 * 6 ** 2))
    blob = lambda y0, x0, A: A * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 1.5 ** 2))
    c = np.log10(host() + blob(75, 30, 3e14) + blob(25, 70, 2e14))
    core = 1 - 0.3 * np.exp(-r ** 2 / 8)
    s = np.log10(host() * core + blob(77, 27, 3e14) + 0.7 * blob(26, 74, 2e14))  # 2nd subhalo moved and lost 30%
    sa, tab = align(c, s)
    print("peaks (y, x, amp, dy, dx, ncc):\n", np.round(tab, 2))
    for (y0, x0) in ((75, 30), (25, 70)):
        dd = np.hypot(yy - y0, xx - x0) < 3
        print(f"  subhalo at ({y0},{x0}) r<3 mass change: before {np.log10((10**s)[dd].sum()/(10**c)[dd].sum()):+.3f} after {np.log10((10**sa)[dd].sum()/(10**c)[dd].sum()):+.3f}")
    print(f"  centre r<1: before {np.log10((10**s)[r<1].sum()/(10**c)[r<1].sum()):+.3f} after {np.log10((10**sa)[r<1].sum()/(10**c)[r<1].sum()):+.3f}; total mass ratio {(10**sa).sum()/(10**s).sum():.5f}")
    print("  expected: first ~0, second ~log10(0.7 + host share) < 0, centre unchanged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--example", type=int, default=1951); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); raise SystemExit
    OUT = f"{HERE}/runs/peak_align"; os.makedirs(OUT, exist_ok=True)
    D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
    x, y, xi, sig, sh = (D[k] for k in ("x", "y", "xi", "sigma", "shift"))
    pkr = np.load(f"{HERE}/data/bahamas_disturbance.npz")["second_peak_ratio"]
    ev = np.where((np.abs(sh).max(1) <= 2) & (sig > 0))[0]
    YA = np.empty((len(ev), n, n), np.float32); T = []
    for k, i in enumerate(ev):
        YA[k], tab = align(x[xi[i]].astype(np.float64), y[i].astype(np.float64))
        T += [(k, *t) for t in tab]
        if k % 1000 == 0: print(f"  {k}/{len(ev)}", flush=True)
    T = np.array(T)  # k, y, x, amp, dy, dx, ncc
    np.savez_compressed(f"{HERE}/data/peak_align.npz", pair=ev, y_aligned=YA, peaks=T,
                        peak_cols=np.array(["k", "y", "x", "amp", "dy", "dx", "ncc"]))
    res = {}; Ssig = sig[ev]
    for s_ in (0.1, 0.3, 1.0):
        R = {}; m = Ssig == s_; ks = np.where(m)[0]
        Tm = T[np.isin(T[:, 0], ks)]; acc = Tm[:, 6] >= CMIN; u = np.hypot(Tm[acc, 4], Tm[acc, 5])
        R["peaks_per_map"] = float(len(Tm) / m.sum()); R["matched_frac"] = float(acc.mean())
        R["shift_median_px"] = float(np.median(u)); R["shift_p90_px"] = float(np.percentile(u, 90)); R["ncc_median"] = float(np.median(Tm[:, 6]))
        # per-peak mass change at the CDM peak, before / after, apertures 1 and 3 px
        for tag, Ymap in (("before", y[ev]), ("after", YA)):
            v1, v3 = [], []
            for k, py, px in Tm[acc][:, :3].astype(int):
                c = 10.0 ** x[xi[ev[k]]].astype(float); t = 10.0 ** Ymap[k].astype(float); d = np.hypot(yy - py, xx - px)
                v1.append(np.log10(t[d < 1.01].sum() / c[d < 1.01].sum())); v3.append(np.log10(t[d < 3].sum() / c[d < 3].sum()))
            R[f"peak_r1_{tag}"] = float(np.median(v1)); R[f"peak_r3_{tag}"] = float(np.median(v3)); R[f"peak_r1_{tag}_std"] = float(np.std(v1))
        # unmatched peaks (lost / unrecognisable in SIDM)
        v1 = []
        for k, py, px in Tm[~acc][:, :3].astype(int):
            c = 10.0 ** x[xi[ev[k]]].astype(float); t = 10.0 ** y[ev[k]].astype(float); d = np.hypot(yy - py, xx - px)
            v1.append(np.log10(t[d < 3].sum() / c[d < 3].sum()))
        R["unmatched_peak_r3"] = float(np.median(v1)) if v1 else None
        # second-peak subcluster table, as before
        for tag, Ymap in (("before", y[ev]), ("after", YA)):
            sub = {q: [] for q in (1, 2, 4, 8)}; cen = []
            for k in ks[pkr[xi[ev[ks]]] >= 0.5]:
                c = 10.0 ** x[xi[ev[k]]].astype(float); t = 10.0 ** Ymap[k].astype(float)
                L = gaussian_filter(c, 2); mx = (maximum_filter(L, 9) == L) & (r > 8) & (r < 42)
                if not mx.any(): continue
                iy, ix = np.unravel_index(np.argmax(np.where(mx, L, 0)), L.shape); dd = np.hypot(yy - iy, xx - ix)
                for q in sub: sub[q].append(np.log10(t[dd < q].sum() / c[dd < q].sum()))
                cen.append(np.log10(t[r < 1].sum() / c[r < 1].sum()))
            R[f"subcluster_{tag}"] = {q: float(np.median(v)) for q, v in sub.items()}; R[f"centre_r1_{tag}"] = float(np.median(cen))
        d0 = (y[ev[m]] - x[xi[ev[m]]])[:, 5:95, 5:95]; d1 = (YA[m] - x[xi[ev[m]]])[:, 5:95, 5:95]
        R["rms"] = [float(d0.std()), float(d1.std())]; R["rms_smooth4"] = [float(gaussian_filter(d0, (0, 4, 4)).std()), float(gaussian_filter(d1, (0, 4, 4)).std())]
        R["total_mass_ratio_after_before"] = float(np.median((10.0 ** YA[m].astype(float)).sum((1, 2)) / (10.0 ** y[ev[m]].astype(float)).sum((1, 2))))
        res[str(s_)] = R
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    for s_ in ("0.1", "0.3", "1.0"):
        R = res[s_]
        print(f"sigma {s_}: {R['peaks_per_map']:.1f} CDM subhalo peaks/map, matched {R['matched_frac']:.0%} (NCC median {R['ncc_median']:.2f}); shift median {R['shift_median_px']:.2f} px, 90% {R['shift_p90_px']:.2f}")
        print(f"   matched peaks, mass change at CDM peak: r<1 {R['peak_r1_before']:+.3f} -> {R['peak_r1_after']:+.3f} (scatter {R['peak_r1_before_std']:.3f} -> {R['peak_r1_after_std']:.3f});  r<3 {R['peak_r3_before']:+.3f} -> {R['peak_r3_after']:+.3f}")
        print(f"   unmatched peaks r<3: {R['unmatched_peak_r3']:+.3f}")
        print(f"   big subcluster: before {' '.join(f'r<{q}:{v:+.3f}' for q, v in R['subcluster_before'].items())}")
        print(f"                   after  {' '.join(f'r<{q}:{v:+.3f}' for q, v in R['subcluster_after'].items())};  centre r<1 {R['centre_r1_before']:+.3f} -> {R['centre_r1_after']:+.3f}")
        print(f"   rms delta {R['rms'][0]:.3f} -> {R['rms'][1]:.3f}; 4px-smoothed {R['rms_smooth4'][0]:.3f} -> {R['rms_smooth4'][1]:.3f}; total mass after/before {R['total_mass_ratio_after_before']:.5f}")
    # example
    k = np.where((xi[ev] == a.example) & (sig[ev] == 1.0))[0]
    if len(k):
        k = k[0]; i = ev[k]; Tk = T[T[:, 0] == k]
        fig, g = plt.subplots(1, 4, figsize=(18, 4.9), constrained_layout=True)
        lo, hi = np.percentile(x[xi[i]], [1, 99.9])
        g[0].imshow(x[xi[i]], cmap="magma", vmin=lo, vmax=hi, origin="lower"); g[0].set_title("CDM + subhalo peaks (o matched, x not)")
        mt = Tk[:, 6] >= CMIN
        g[0].plot(Tk[mt, 2], Tk[mt, 1], "o", mfc="none", mec="c", ms=7); g[0].plot(Tk[~mt, 2], Tk[~mt, 1], "x", c="w", ms=6)
        g[1].imshow(y[i] - x[xi[i]], cmap="RdBu_r", vmin=-.3, vmax=.3, origin="lower"); g[1].set_title("Δ before (σ=1)")
        g[1].quiver(Tk[mt, 2], Tk[mt, 1], Tk[mt, 5], Tk[mt, 4], angles="xy", scale_units="xy", scale=0.5, color="k", width=0.004)
        g[1].set_xlim(0, 99); g[1].set_ylim(0, 99)
        im = g[2].imshow(YA[k] - x[xi[i]], cmap="RdBu_r", vmin=-.3, vmax=.3, origin="lower"); g[2].set_title("Δ after per-subhalo alignment")
        g[3].imshow(YA[k] - y[i], cmap="PuOr", vmin=-.3, vmax=.3, origin="lower"); g[3].set_title("aligned − original SIDM (what moved)")
        fig.colorbar(im, ax=g[1:], shrink=0.8, label="dex")
        for q in g: q.set_xticks([]); q.set_yticks([])
        fig.suptitle(f"per-subhalo alignment, BAHAMAS row {a.example}, σ=1 (arrows ×2: CDM → SIDM subhalo position)")
        fig.savefig(f"{OUT}/example_{a.example}.png", dpi=100)
    print("wrote", OUT)

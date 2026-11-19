"""Probabilistic CDM -> SIDM generators on BAHAMAS DM-only maps (see dm_common.py); writes NDRAW draws per test pair.

--method gauss    : mean + a(r, S_c) * g, g heavy-tailed stationary noise (as dm_distribution.py)
--method displace : the CDM small-scale structure (after the rank map) is moved by a random smooth displacement field
                    u(x), drawn with the radial rms and power spectrum of the displacements measured between real CDM and
                    SIDM runs (data/warp_u.npz, warp_test.py; tapered to 0 at the centre where the core is aligned), plus
                    extra noise a_x(r, S_c) g carrying only the variance the displacement does not explain
                    (a_x^2 = max(a_true^2 - a_disp^2, 0)), with the stationary filter tuned so the total power of
                    SIDM - mean matches training.
Output runs/dm_<method>/draws.npz: draws (NDRAW, Ntest, 96, 96) float16 of the SIDM map, k (pairs index), mu.
usage: python3 dm_generators.py --method gauss|displace [--ndraw 19] [--uscale 1]  (displace, uscale != 1 -> runs/dm_displace_u<uscale>)
"""
import os, argparse, time
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import dm_common as C, two_scale as T2
ap = argparse.ArgumentParser(); ap.add_argument("--method", choices=("gauss", "displace"), required=True)
ap.add_argument("--ndraw", type=int, default=19)
ap.add_argument("--uscale", type=float, default=1.0, help="scale the measured displacement rms (displace only)"); a = ap.parse_args()
OUT = f"{C.HERE}/runs/dm_{a.method}" + (f"_u{a.uscale:g}" if a.method == "displace" and a.uscale != 1 else ""); os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(7); t0 = time.time()
S, D = C.load(); tr, te = S["train"], S["test"]; print(f"train {len(tr['k'])}, test {len(te['k'])}, {time.time() - t0:.0f}s", flush=True)
SEDG = C.sedges(tr); e_tr = tr["s"] - tr["mu"]
yy, xx = np.indices((C.N, C.N)).astype(np.float64)

if a.method == "displace":
    W = np.load(f"{C.HERE}/data/warp_u.npz"); U = W["u"][::4, :, 2:98, 2:98].astype(np.float32)  # (n, 2, 96, 96) px
    urms = np.sqrt(np.bincount(C.T2.RB.ravel(), (U ** 2).mean((0, 1)).ravel()) / np.bincount(C.T2.RB.ravel()))  # per component, by radius
    UR = a.uscale * urms[T2.RB]; Un = U / np.maximum(urms[T2.RB], 1e-3)[None, None]
    UF = np.sqrt(C.praw(Un.reshape(-1, C.N, C.N))[C.KB]); UF[0, 0] = 0
    print("displacement rms per component (px) at r = 5/10/20/40 px:", np.round(urms[[5, 10, 20, 40]], 2), flush=True)
    def displace(P, n):
        out = np.empty((n, len(P["c"]), C.N, C.N), np.float32)
        for d in range(n):
            u = C.gfield(rng, (len(P["c"]), 2, C.N, C.N), UF) * UR[None, None]
            for i in range(len(P["c"])):
                out[d, i] = map_coordinates(P["Sm"][i].astype(np.float64), [yy - u[i, 0], xx - u[i, 1]], order=1, mode="reflect")
        return P["Lc"][None] + P["dL"][None] + out
    sub = C.sub(tr, np.arange(0, len(tr["k"]), 4)); base_sub = displace(sub, 1)[0]
    a_true = C.amp_table(sub["s"] - sub["mu"], sub, SEDG); a_disp = C.amp_table(base_sub - sub["mu"], sub, SEDG)
    amp = np.sqrt(np.clip(a_true ** 2 - a_disp ** 2, 0, None)); marg = None
    print("rms true resid vs displacement-only (sigma=1, by radial bin, median over S_c):",
          np.round(np.median(a_true[2], 1), 3), np.round(np.median(a_disp[2], 1), 3), flush=True)
else:
    amp = C.amp_table(e_tr, tr, SEDG); sub = C.sub(tr, np.arange(0, len(tr["k"]), 4)); base_sub = sub["mu"][None][0]
    QL = np.linspace(0.0005, 0.9995, 400); marg = (QL, np.quantile((e_tr / C.a_map(amp, tr, SEDG))[::3], QL))
    displace = lambda P, n: np.broadcast_to(P["mu"][None], (n,) + P["mu"].shape)
A_sub = C.a_map(amp, sub, SEDG)
FILT = np.sqrt(C.praw((e_tr / np.maximum(C.a_map(amp, tr, SEDG), 1e-3))[::5])[C.KB]); FILT[0, 0] = 0
target = C.praw(sub["s"] - sub["mu"])
for it in range(4):
    smp = base_sub + A_sub * C.gfield(rng, A_sub.shape, FILT, marg)
    ratio = target / np.maximum(C.praw(smp - sub["mu"]), 1e-12); ratio[0] = 1; ratio = np.clip(ratio, 0.01, 100)
    FILT = FILT * np.sqrt(ratio)[C.KB]; FILT[0, 0] = 0
    print(f"spectral iteration {it}: max |log10 ratio| {np.abs(np.log10(ratio[1:C.N // 2])).max():.3f}", flush=True)
A_te = C.a_map(amp, te, SEDG)
draws = displace(te, a.ndraw) + A_te[None] * np.stack([C.gfield(rng, A_te.shape, FILT, marg) for _ in range(a.ndraw)])
np.savez(f"{OUT}/draws.npz", draws=draws.astype(np.float16), k=te["k"], amp=amp, filt=FILT)
print(f"wrote {OUT}/draws.npz {draws.shape}, {time.time() - t0:.0f}s", flush=True)

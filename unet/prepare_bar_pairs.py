"""Stars and gas CDM -> SIDM pairs for BAHAMAS, row-for-row aligned with data/pairs_dm.npz (same xi, sigma, split,
re-centring shift applied to the target).

Maps: particle-binned Sigma_star and Sigma_gas, |los| < 5 pMpc (deep5; baryon_maps/<model>/snap_<snap>/los<k>.npz from
velmaps/bahamas/bahamas_baryon_maps.py), stamp orientation (orientation.json), smoothed SMOOTH px like the DM pairs,
stars log10(Sigma + SOFT_S) (soft floor: one smoothed star particle peaks at ~2e11, so single particles stay visible
but their faint wings and empty pixels converge smoothly to 11, no spike), gas log10(max(Sigma, FLOOR_G)).
Output data/pairs_bar.npz: xs, xg (CDM, (Nx,100,100)), ys, yg (SIDM, (P,100,100)) log10 Msun/pMpc^2, ok_bar (P,) bool,
plus xi, sigma, shift, split, redshift, projection, log10_M200, match_id from pairs_dm.npz.
usage: python prepare_bar_pairs.py
"""
import os, json, pickle
import numpy as np
from scipy.ndimage import gaussian_filter as G
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"; BM = f"{IMG}/baryon_maps"
SMOOTH, SOFT_S, FLOOR_G = 1.5, 1e11, 1e11
SNAP = {0.0: "032", 0.125: "031", 0.25: "030", 0.375: "029"}
MOD = {0.0: ("CDM+baryons", "cdm"), 0.1: ("SIDM0.1+baryons", "sidm0.1"), 0.3: ("SIDM0.3+baryons", "sidm0.3"), 1.0: ("SIDM1+baryons", "sidm1")}
O = json.load(open(f"{HERE}/../velmaps/bahamas/orientation.json"))
meta, _ = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb")); keep = np.where(meta["mask"][:, 0])[0]
D = np.load(f"{HERE}/data/pairs_dm.npz", allow_pickle=True)
assert (D["match_id"] == meta["match_id"][keep]).all()
cache = {}


def bar_maps(k, s):
    z, P = float(meta["redshift"][k]), int(meta["projection"][k]); model, key = MOD[s]; fof = int(meta[f"fof_{key}"][k])
    if fof < 0: return None
    o = O[f"projection_{P}"]; f = f"{BM}/{model}/snap_{SNAP[z]}/los{o['los']}.npz"
    if f not in cache:
        d = np.load(f); cache[f] = (d["maps"][:, [2, 0]].copy(), {int(v): i for i, v in enumerate(d["fof"])}); print("loaded", f, flush=True)
    M, ix = cache[f]
    if fof not in ix: return None
    out = []
    for c in (0, 1):
        m = np.rot90(G(M[ix[fof], c].astype(np.float64), SMOOTH), o["rot"]); m = m[:, ::-1] if o["flip"] else m
        out.append(np.log10(m + SOFT_S if c == 0 else np.maximum(m, FLOOR_G)).astype(np.float32))
    return out


nan = np.full((100, 100), np.nan, np.float32)
X = [bar_maps(k, 0.0) or [nan, nan] for k in keep]; xs = np.stack([a for a, _ in X]); xg = np.stack([b for _, b in X])
xi, sig, sh = D["xi"], D["sigma"], D["shift"]; ys = np.empty((len(xi), 100, 100), np.float32); yg = np.empty_like(ys)
for p in range(len(xi)):
    if sig[p] == 0: ys[p], yg[p] = xs[xi[p]], xg[xi[p]]; continue
    m = bar_maps(keep[xi[p]], float(sig[p]))
    ys[p], yg[p] = (nan, nan) if m is None else (np.roll(q, tuple(sh[p]), (0, 1)) for q in m)
ok = np.isfinite(xs[xi]).all((1, 2)) & np.isfinite(ys).all((1, 2)) & D["ok"]
print(f"{ok.sum()}/{len(ok)} pairs with DM, star and gas maps;", {s: int((ok & (sig == s)).sum()) for s in (0.1, 0.3, 1.0)})
np.savez(f"{HERE}/data/pairs_bar.npz", xs=xs, xg=xg, ys=ys, yg=yg, ok_bar=ok,
         **{k: D[k] for k in ("xi", "sigma", "shift", "split", "redshift", "projection", "log10_M200", "match_id")})
print("wrote data/pairs_bar.npz")

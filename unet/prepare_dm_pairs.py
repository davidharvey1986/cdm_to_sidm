"""Dark-matter-only CDM -> SIDM pairs for BAHAMAS, row-for-row aligned with data/pairs.npz.

Maps: log10 Sigma_DM from the dark-matter velocity-moment files (velmaps_v2_imaged, W0 of the deep5 selection,
|los| < 5 pMpc = the depth of the total-mass stamps), each run centred on its own catalogue centre, in stamp
orientation (orientation.json), W0 smoothed SMOOTH px (BAHAMAS: m_dm = 5.5e9 Msun, ~1 particle per pixel at 1 Mpc),
floored at 1e11 Msun/pMpc^2. Same pairs as pairs.npz (same xi, sigma, split, re-centring shift applied to the target).
Output data/pairs_dm.npz with the pairs.npz keys (x, y, xi, sigma, shift, split, redshift, projection, log10_M200,
match_id, x_Mpc_h) plus ok (P,) bool = both maps present.
usage: python3 prepare_dm_pairs.py
"""
import os, json, pickle
import numpy as np
from scipy.ndimage import gaussian_filter as G
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"; V = f"{IMG}/velmaps_v2_imaged"
SMOOTH, FLOOR = 1.5, 1e11
SNAP = {0.0: "032", 0.125: "031", 0.25: "030", 0.375: "029"}
MOD = {0.0: ("CDM+baryons", "cdm"), 0.1: ("SIDM0.1+baryons", "sidm0.1"), 0.3: ("SIDM0.3+baryons", "sidm0.3"), 1.0: ("SIDM1+baryons", "sidm1")}
O = json.load(open(f"{HERE}/../velmaps/bahamas/orientation.json"))
meta, _ = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb")); keep = np.where(meta["mask"][:, 0])[0]
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
assert (D["match_id"] == meta["match_id"][keep]).all()
cache = {}
def dm_map(k, s):
    z, P = float(meta["redshift"][k]), int(meta["projection"][k]); model, key = MOD[s]; fof = int(meta[f"fof_{key}"][k])
    if fof < 0: return None
    o = O[f"projection_{P}"]; f = f"{V}/{model}/snap_{SNAP[z]}/los{o['los']}.npz"
    if f not in cache:
        d = np.load(f); cache[f] = (d["moments"][:, 0].copy(), {int(v): i for i, v in enumerate(d["fof"])}); print("loaded", f, flush=True)
    W0, ix = cache[f]
    if fof not in ix: return None
    m = np.rot90(G(W0[ix[fof]].astype(np.float64), SMOOTH), o["rot"]); m = m[:, ::-1] if o["flip"] else m
    return np.log10(np.maximum(m, FLOOR)).astype(np.float32)
nan = np.full((100, 100), np.nan, np.float32)
x = np.stack([(lambda m: nan if m is None else m)(dm_map(k, 0.0)) for k in keep])
xi, sig, sh = D["xi"], D["sigma"], D["shift"]; y = np.empty((len(xi), 100, 100), np.float32)
for p in range(len(xi)):
    if sig[p] == 0: y[p] = x[xi[p]]; continue
    m = dm_map(keep[xi[p]], float(sig[p])); y[p] = nan if m is None else np.roll(m, tuple(sh[p]), (0, 1))
ok = np.isfinite(x[xi]).all((1, 2)) & np.isfinite(y).all((1, 2))
print(f"{ok.sum()}/{len(ok)} pairs with both DM maps;", {s: int((ok & (sig == s)).sum()) for s in (0.1, 0.3, 1.0)})
np.savez(f"{HERE}/data/pairs_dm.npz", x=x, y=y, ok=ok, **{k: D[k] for k in ("xi", "sigma", "shift", "split", "redshift", "projection", "log10_M200", "match_id", "x_Mpc_h")})
print("wrote data/pairs_dm.npz")

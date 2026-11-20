"""Training file for the gas flow (step 3 of the DM -> stars -> gas chain): data/pairs_gas.npz.
  x  (Nx, 100, 100)    CDM gas, log10 max(Sigma_gas, 1e11)                (pairs_bar.npz xg)
  y  (P, 100, 100)     SIDM gas, same units, re-centring shift applied     (pairs_bar.npz yg)
  cx (Nx, 2, 100, 100) CDM dark matter (log10 Sigma), CDM stars (log10(Sigma + 1e11))
  cy (P, 2, 100, 100)  SIDM - CDM dark matter, SIDM - CDM stars (dex); at test time from the DM-flow and stars-flow draws
  ok (P,), plus xi, sigma, shift, split, redshift, projection, log10_M200, match_id.
train_flow.py --model A --data pairs_gas.npz learns p(SIDM gas - CDM gas | CDM gas, CDM DM, CDM stars, dDM, dstars, sigma, z).
usage: python prepare_gas_pairs.py
"""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
B = np.load(f"{HERE}/data/pairs_bar.npz", allow_pickle=True); Dd = np.load(f"{HERE}/data/pairs_dm.npz", allow_pickle=True)
assert (B["xi"] == Dd["xi"]).all() and (B["sigma"] == Dd["sigma"]).all()
xd, yd, xs, ys, xi = Dd["x"], Dd["y"], B["xs"], B["ys"], Dd["xi"]
cx = np.stack([xd, xs], 1); cy = np.stack([yd - xd[xi], ys - xs[xi]], 1).astype(np.float32)
np.savez(f"{HERE}/data/pairs_gas.npz", x=B["xg"], y=B["yg"], cx=cx, cy=cy, ok=B["ok_bar"] & Dd["ok"],
         **{k: B[k] for k in ("xi", "sigma", "shift", "split", "redshift", "projection", "log10_M200", "match_id")})
print("wrote data/pairs_gas.npz")

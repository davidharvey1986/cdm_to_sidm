"""Training file for the stars flow (step 2 of the DM -> stars -> gas chain): data/pairs_stars.npz.
  x  (Nx, 100, 100)    CDM stars, log10(Sigma_star + 1e11)          (pairs_bar.npz xs)
  y  (P, 100, 100)     SIDM stars, same units, re-centring shift applied (pairs_bar.npz ys)
  cx (Nx, 1, 100, 100) CDM dark matter, log10 Sigma_DM                (pairs_dm.npz x)
  cy (P, 1, 100, 100)  SIDM - CDM dark matter, dex                    (pairs_dm.npz y - x); at test time from a DM-flow draw
  ok (P,) both baryon and DM maps present; plus xi, sigma, shift, split, redshift, projection, log10_M200, match_id.
train_flow.py --model A --data pairs_stars.npz then learns p(SIDM stars - CDM stars | CDM stars, CDM DM, Delta DM, sigma, z).
usage: python prepare_stars_pairs.py
"""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
B = np.load(f"{HERE}/data/pairs_bar.npz", allow_pickle=True); Dd = np.load(f"{HERE}/data/pairs_dm.npz", allow_pickle=True)
assert (B["xi"] == Dd["xi"]).all() and (B["sigma"] == Dd["sigma"]).all()
xd, yd, xi = Dd["x"], Dd["y"], Dd["xi"]
np.savez(f"{HERE}/data/pairs_stars.npz", x=B["xs"], y=B["ys"], cx=xd[:, None], cy=(yd - xd[xi])[:, None], ok=B["ok_bar"] & Dd["ok"],
         **{k: B[k] for k in ("xi", "sigma", "shift", "split", "redshift", "projection", "log10_M200", "match_id")})
print("wrote data/pairs_stars.npz")

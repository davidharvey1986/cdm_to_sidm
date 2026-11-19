"""Hybrid targets for the DM-only flow: the two-scale converter sets the mean change, the flow models what is left.

Writes data/pairs_dm_hyb.npz = data/pairs_dm.npz with the SIDM map y replaced, inside the 96x96 crop, by
    y_hyb = y - (mu - c)          (mu = converter mean prediction, c = CDM map; dm_common.load)
so train_flow.py --data pairs_dm_hyb.npz learns p(y - mu | c, sigma, z): the residual about the converter mean.
A draw is then  c + (mu - c) + flow sample = mu + flow sample  (dm_flow_draws.py --hybrid).
Pairs outside the dm_common selection (shift > 2 px or sigma = 0) are left unchanged; train_flow.py drops them too.
The converter (runs/dm_distribution/template.json) was fitted on the train split only.
usage: python3 prepare_dm_hybrid.py
"""
import numpy as np
import dm_common as C
S, D = C.load(); y = D["y"].copy(); n = 0
for lab, P in S.items():
    y[P["k"], 2:98, 2:98] = P["s"] - (P["mu"] - P["c"]); n += len(P["k"])
    r = P["s"] - P["mu"]; print(f"{lab}: {len(P['k'])} pairs, residual rms {r.std():.4f} dex, core mean {r[:, 44:52, 44:52].mean():+.4f}")
np.savez(f"{C.HERE}/data/pairs_dm_hyb.npz", **{k: D[k] for k in D.files if k != "y"}, y=y)
print(f"wrote data/pairs_dm_hyb.npz ({n} pairs re-targeted)")

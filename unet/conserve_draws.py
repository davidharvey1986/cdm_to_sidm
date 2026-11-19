"""Mass-conserving version of a set of DM draws: runs/dm_<name>/draws.npz -> runs/dm_<name>_mc/draws.npz, every draw
rescaled (a constant shift in log Sigma) so its mass within 900 kpc equals the CDM map's times a ratio drawn from the
train-split SIDM/CDM distribution (dm_common.conserve; --exact: exactly the CDM mass -> runs/dm_<name>_mcx).
usage: python conserve_draws.py flow_hyb [--exact]
"""
import os, sys
import numpy as np
import dm_common as C
n = sys.argv[1]; EX = "--exact" in sys.argv; S, _ = C.load(); te = S["test"]; d = np.load(f"{C.HERE}/runs/dm_{n}/draws.npz")
assert (d["k"] == te["k"]).all()
X = d["draws"].astype(np.float32); Y = C.conserve(X, te["c"], None if EX else te["sigma"])
sh = (Y - X)[..., 0, 0]; print(f"{n}: applied shifts rms {sh.std():.4f} dex, max |shift| {np.abs(sh).max():.3f}")
od = f"{C.HERE}/runs/dm_{n}_mc" + ("x" if EX else ""); os.makedirs(od, exist_ok=True)
np.savez(f"{od}/draws.npz", **{k: d[k] for k in d.files if k != "draws"}, draws=Y.astype(np.float16), conserved="R<900kpc mass = CDM" + ("" if EX else " x train-split SIDM/CDM ratio draw"))
print("wrote", od + "/draws.npz")

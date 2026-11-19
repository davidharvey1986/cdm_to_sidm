"""Which box axis / image orientation of the COSMA velocity maps matches each stamp projection_P?
Compares log DM surface density (W0, fof) with the stamp total-mass map for the test clusters (CDM+baryons, z=0)
under every line-of-sight axis and the 8 flips/rotations. usage: python3 check_orientation.py"""
import os, pickle
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
HERE = os.path.dirname(os.path.abspath(__file__)); T = f"{HERE}/test_output"
st = pd.read_csv(f"{HERE}/../../bahamas/bahamas_stamps.csv")
_, im = pickle.load(open("/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas/bahamas.pkl", "rb"))
L = {k: np.load(f"{T}/los{k}.npz") for k in range(3)}
fofs = L[0]["fof"]
D4 = {f"rot{r}{'_flip' if f else ''}": (r, f) for r in range(4) for f in (0, 1)}
def tf(m, r, f):
    m = np.rot90(m, r); return m[:, ::-1] if f else m
res = {}
for p in (0, 1, 2):
    for i, fof in enumerate(fofs):
        row = st[(st.model == "CDM+baryons") & (st.redshift == 0) & (st.projection == p) & (st.fof == fof)]
        if not len(row): continue
        S = np.log10(np.asarray(im[int(row.idx_combined.iloc[0])])[..., 0])
        for k in range(3):
            W = np.log10(gaussian_filter(L[k]["moments"][i, 0].astype(float), 1) + 1e10)
            for nm, (r, f) in D4.items():
                c = np.corrcoef(tf(W, r, f).ravel(), S.ravel())[0, 1]
                res.setdefault((p, k, nm), []).append(c)
for p in (0, 1, 2):
    best = sorted(((np.mean(v), k, nm) for (pp, k, nm), v in res.items() if pp == p), reverse=True)[:3]
    print(f"projection_{p}: best " + " | ".join(f"los{k} {nm} corr {c:.3f}" for c, k, nm in best))

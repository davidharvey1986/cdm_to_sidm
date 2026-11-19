"""Darkskies training pairs in the same format as data/pairs.npz -> data/darkskies_pairs.npz

Rows of ../halo_database.csv with matched == True (unfixable6 excluded), 20 kpc 100x100 maps, log10 Sigma
(Msun/pMpc^2). One x per (halo, projection, snap) CDM map; one pair per SIDM model (sigma 0.01/0.05/0.1/0.2).
SIDM map re-centred on the CDM map by the integer cross-correlation shift (|shift| <= 12 px), stored in `shift`.
Split by halo (all projections/snaps/sigmas of a halo together): 85 train / 15 val halos, seeded.
No test split: the test set for this direction is BAHAMAS.
usage: python3 prepare_darkskies.py
"""
import os, pickle
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ZMAP = {7: 0.375, 8: 0.25, 9: 0.125, 10: 0.0}
db = pd.read_csv(f"{ROOT}/halo_database.csv")
bad = set("/".join(l.split()[:3]) for l in open(f"{ROOT}/unfixable6.txt") if l.strip().startswith("SIDM"))
db = db[db.matched & db.sidm_exists]
db = db[~db.apply(lambda r: f"{r.sidm_model}/{r.snap}/{r.halo}" in bad, axis=1)].reset_index(drop=True)
print(len(db), "matched SIDM rows;", len(bad), "unfixable excluded")


def best_shift(A, B, M=12):
    A, B = A - A.mean(), B - B.mean()
    xc = np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))).real
    d = (np.arange(100) + 50) % 100 - 50; ok = np.abs(d) <= M
    xc = np.where(ok[:, None] & ok[None, :], xc, -np.inf)
    i, j = np.unravel_index(np.argmax(xc), xc.shape); return int(d[i]), int(d[j])


load = lambda p: np.log10(np.asarray(pickle.load(open(p, "rb"))[0])[0]).astype(np.float32)
halos = np.array(sorted(db.halo.unique()))
rng = np.random.default_rng(0); val_h = set(rng.choice(halos, 15, replace=False))
xrow, X, Z, M, HALO, PROJ, SPL = {}, [], [], [], [], [], []
Y, XI, S, SH = [], [], [], []
for r in db.itertuples():
    key = r.cdm_path
    if key not in xrow:
        c = load(key)
        if not np.isfinite(c).all():
            continue
        xrow[key] = len(X); X.append(c); Z.append(ZMAP[r.snap]); M.append(np.log10(r.M200c_cdm * 1e10))
        HALO.append(r.halo); PROJ.append(r.projection); SPL.append("val" if r.halo in val_h else "train")
    t = load(r.sidm_path)
    if not np.isfinite(t).all():
        continue
    i = xrow[key]; dy, dx = best_shift(X[i], t)
    Y.append(np.roll(t, (dy, dx), (0, 1))); XI.append(i); S.append(r.sigma); SH.append((dy, dx))
S, SH, XI = np.array(S), np.array(SH), np.array(XI)
np.savez(f"{HERE}/data/darkskies_pairs.npz", x=np.stack(X), y=np.stack(Y).astype(np.float32), xi=XI, sigma=S, shift=SH,
         split=np.array(SPL), redshift=np.array(Z), projection=np.array(PROJ), log10_M200=np.array(M, np.float32),
         match_id=np.array(HALO))
sp = np.array(SPL)[XI]; k = np.abs(SH).max(1) <= 2
print(f"{len(X)} CDM maps, {len(S)} pairs; |shift|<=2: {k.sum()}; train/val pairs (cut) {(k & (sp == 'train')).sum()}/{(k & (sp == 'val')).sum()}")
print({s: int((k & (S == s)).sum()) for s in np.unique(S)})

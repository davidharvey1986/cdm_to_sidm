"""Gas hybrid: the converter sets the mean gas change, the gas flow models what is left (as flowA_dm_hyb for the DM).

Converter mean (gas_converter.py, fitted on the train split): ring-mean Delta = log SIDM gas - log CDM gas per pair,
least squares per ring (0-45, 20 kpc) and sigma with z as categorical offsets and linear in log M200 - 14.3.
Writes data/gas_hyb_offset.npz: off_ring (n_pairs, 48) = the converter mean change per ring for every pair (0 at sigma = 0;
rings >= 46 take ring 45). train_flow.py --offset gas_hyb_offset.npz subtracts off_ring[RB] from the SIDM map after any
augmentation, so the flow learns p(Delta - mean | CDM gas, DM, stars, dDM, dstars, sigma, z); a draw is
CDM gas + mean + flow sample (gas_flow_draws.py, read from the model's norm["offset"]), hard floor re-applied.
The mean is a log offset, so it is scale-free (unchanged by rescaling the gas or the stars).
usage: python prepare_gas_hybrid.py
"""
import numpy as np
import dm_common as C, two_scale as T2
NR = 46; RB = T2.RB; IN = RB < NR; cnt = np.bincount(RB[IN], minlength=NR)
ringm = lambda d: np.stack([np.bincount(RB[IN], v[IN], NR) for v in d]) / cnt
D = np.load(f"{C.HERE}/data/pairs_gas.npz", allow_pickle=True)
xi, sig, split, zz, lm = D["xi"], D["sigma"], D["split"], D["redshift"], D["log10_M200"]
ok = D["ok"] & (np.abs(D["shift"]).max(1) <= 2); tr = np.where(ok & (sig > 0) & (split[xi] == "train"))[0]; ZS = np.unique(zz)
X, Y = D["x"], D["y"]; off = np.zeros((len(sig), 48), np.float32)
design = lambda j: np.column_stack([np.stack([zz[xi[j]] == z for z in ZS], 1).astype(float), lm[xi[j]] - 14.3])
for s in C.SIG:
    idx = tr[sig[tr] == s]; rp = ringm((Y[idx][:, 2:98, 2:98] - X[xi[idx]][:, 2:98, 2:98]).astype(np.float64))
    coef = np.linalg.lstsq(design(idx), rp, rcond=None)[0]
    allp = np.where(np.isclose(sig, s))[0]; m = design(allp) @ coef
    off[allp, :NR] = m; off[allp, NR:] = m[:, -1:]
    print(f"SIDM{s:g}: fitted on {len(idx)} train pairs; mean offset R<40 kpc {m[:, :2].mean():+.4f}, 100 kpc {m[:, 5].mean():+.4f}, 500 kpc {m[:, 25].mean():+.4f} dex")
np.savez(f"{C.HERE}/data/gas_hyb_offset.npz", off_ring=off, data="pairs_gas.npz"); print("wrote data/gas_hyb_offset.npz")

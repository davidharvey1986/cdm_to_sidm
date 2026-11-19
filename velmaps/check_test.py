"""Check the velocity-map test output (CDM snap 7 halo_00000001) against the nn100 total-mass map and the
VELOCIraptor aperture dispersions. usage: python3 check_test.py  -> test_output/check_test.png"""
import os, pickle
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
maps, meta = pickle.load(open(f"{D}/halo_00000001_x.pkl", "rb"))
tot, tmeta = pickle.load(open(f"{D}/mass_nn100_halo_00000001_x.pkl", "rb")); tot = np.asarray(tot)[0]
others = {ax: pickle.load(open(f"{D}/halo_00000001_{ax}.pkl", "rb"))[0] for ax in "yz"}
p = meta["props"]; yy, xx = np.indices((100, 100)); R = np.hypot(yy - 49.5, xx - 49.5) * 20
fig, ax = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
kw = dict(origin="lower", interpolation="nearest")
im = ax[0, 0].imshow(np.log10(tot), cmap="magma", **kw); ax[0, 0].set_title("total mass (nn100 map), log Σ"); fig.colorbar(im, ax=ax[0, 0], shrink=0.8)
im = ax[0, 1].imshow(np.log10(maps[0]), cmap="magma", **kw); ax[0, 1].set_title("DM Σ (velocity run), log Σ"); fig.colorbar(im, ax=ax[0, 1], shrink=0.8)
im = ax[0, 2].imshow(maps[1], cmap="RdBu_r", vmin=-800, vmax=800, **kw); ax[0, 2].set_title("mean LOS velocity [km/s]"); fig.colorbar(im, ax=ax[0, 2], shrink=0.8)
im = ax[0, 3].imshow(maps[2], cmap="viridis", vmin=200, vmax=1000, **kw); ax[0, 3].set_title("LOS dispersion, full depth [km/s]"); fig.colorbar(im, ax=ax[0, 3], shrink=0.8)
im = ax[1, 0].imshow(maps[4], cmap="viridis", vmin=200, vmax=1000, **kw); ax[1, 0].set_title("LOS dispersion, ±R200 slab [km/s]"); fig.colorbar(im, ax=ax[1, 0], shrink=0.8)
im = ax[1, 1].imshow(np.log10(maps[0] / tot), cmap="RdBu_r", vmin=-0.3, vmax=0.3, **kw); ax[1, 1].set_title("log DM Σ / total Σ"); fig.colorbar(im, ax=ax[1, 1], shrink=0.8)
rb = (R // 20).astype(int); cnt = np.bincount(rb.ravel())
prof = lambda m: np.bincount(rb.ravel(), np.nan_to_num(m).ravel()) / cnt
r_ = (np.arange(len(cnt)) + 0.5) * 20
ax[1, 2].plot(r_, prof(maps[2]), "k", lw=2, label="x full depth"); ax[1, 2].plot(r_, prof(maps[4]), "k--", label="x slab")
for a_, m_ in others.items(): ax[1, 2].plot(r_, prof(m_[2]), lw=1.2, label=f"{a_} full depth")
for ap_, mk in ((10, "o"), (30, "s"), (50, "^"), (100, "D")):
    ax[1, 2].plot(ap_, p[f"Aperture_veldisp_{ap_}_kpc"], mk, color="r", ms=8, label=f"VR aperture {ap_} kpc" if ap_ == 10 else None)
ax[1, 2].set_xscale("log"); ax[1, 2].set_xlabel("R [kpc]"); ax[1, 2].set_ylabel("σ_los [km/s]"); ax[1, 2].legend(fontsize=8); ax[1, 2].set_title("radial profile vs VELOCIraptor")
ax[1, 3].hist(np.log10(maps[0] / tot).ravel(), 60, color="0.4"); ax[1, 3].set_xlabel("log DM Σ / total Σ"); ax[1, 3].set_title("DM fraction per pixel")
for a_ in ax.ravel()[:6]: a_.set_xticks([]); a_.set_yticks([])
ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])
fig.suptitle(f"velocity-map test: CDM snap 7 halo_00000001 x  (z={meta['redshift']}, M200={p['Mass_200crit']*1e10:.2e}, R200={p['R_200crit']:.2f} pMpc)")
fig.savefig(f"{D}/check_test.png", dpi=85)
print("median DM/total", np.median(maps[0] / tot), " centre sigma", maps[2][48:52, 48:52].mean(), "NaN px", np.isnan(maps[2]).sum())

"""Show remade (nn100, correctly matched) darkskies stamps next to CDM and the old version they replaced.
usage: python3 plot_remade_example.py MODEL/SNAP/halo_XXXXXXXX_P [more ...]"""
import sys, pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
I = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images"
keys = sys.argv[1:] or ["SIDM0.1/9/halo_00016594_x"]
ld = lambda f: pickle.load(open(f, "rb"))
fig, ax = plt.subplots(len(keys), 4, figsize=(16, 4.1 * len(keys)), facecolor="white", squeeze=False)
for i, k in enumerate(keys):
    model, snap, name = k.split("/")
    (c, mc), (o, mo), (n, mn) = (ld(f"{I}/{p}.pkl") for p in (f"darkskies/CDM/{snap}/{name}",
                                 f"darkskies_mismatched_backup/{k}", f"darkskies/{k}"))
    c, o, n = (np.log10(np.asarray(v)[0]) for v in (c, o, n))
    lo, hi = np.percentile(c, 1), c.max()
    cc = lambda a: np.corrcoef(a.ravel(), c.ravel())[0, 1]
    for j, (m, t) in enumerate(((c, "CDM"), (o, f"{model} old (wrong halo), corr {cc(o):.2f}"),
                               (n, f"{model} remade, nn100, corr {cc(n):.2f}"))):
        im = ax[i, j].imshow(m, origin="lower", cmap="magma", vmin=lo, vmax=hi); ax[i, j].set_title(t, fontsize=10)
    fig.colorbar(im, ax=ax[i, 2], fraction=0.046, label="log10 Sigma [Msun/pMpc^2]")
    im = ax[i, 3].imshow(n - c, origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    fig.colorbar(im, ax=ax[i, 3], fraction=0.046, label="dex"); ax[i, 3].set_title("log10(remade / CDM)", fontsize=10)
    ax[i, 0].set_ylabel(f"{name}  snap {snap}  z={mn['redshift']:.3f}\nvr_index {mn['vr_index']}", fontsize=9)
    print(k, "matched_by:", mn["matched_by"], "| old corr", round(cc(o), 3), "new corr", round(cc(n), 3))
    for a in ax[i]: a.set_xticks([]); a.set_yticks([])
fig.suptitle("darkskies 20 kpc stamps (2 pMpc): CDM vs old and remade SIDM", y=1.0)
fig.tight_layout(); fig.savefig("figures/remade_example.png", dpi=110, bbox_inches="tight")

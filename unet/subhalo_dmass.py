"""Change in subhalo mass vs subhalo mass, CDM -> SIDM (BAHAMAS), by abundance matching.
Subhaloes are found independently in each map (subhalo_profiles.py), so they cannot be paired one-to-one; instead,
at each cumulative abundance n(>M) per map, Delta log M = log M_SIDM(n) - log M_CDM(n) (same clusters for both).
Done for sigma = 0.1, 0.3, 1, inside and outside 0.4 Mpc (20 px) from the cluster centre; bands = bootstrap over clusters.
Mass = subhalo excess mass inside 60 kpc (3 px) over the local host.
usage: python3 subhalo_dmass.py  -> figures/subhalo_dmass.png
"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{HERE}/subhalo_profiles.py").read().split("k = np.where(")[0])  # subhaloes(), data
SIG = (0.1, 0.3, 1.0); RSPLIT = 20.0
cat = {}
for s in SIG:
    k = np.where((np.abs(sh).max(1) <= 2) & (sig == s))[0]
    for nm, maps in (("CDM", x[xi[k]]), ("SIDM", y[k])):
        rows = []
        for j, m in enumerate(maps):
            rows += [(j, *t[:2]) for t in subhaloes(m)]
        cat[s, nm] = np.array(rows)
    cat[s, "nmap"] = len(k); print(f"sigma {s}: {len(k)} maps, {len(cat[s, 'CDM'])} CDM / {len(cat[s, 'SIDM'])} SIDM subhaloes")
np.savez_compressed(f"{HERE}/data/subhalo_dmass_cats.npz", **{f"{s}_{n}": v for (s, n), v in cat.items() if n != "nmap"})

Mg = np.linspace(11.0, 12.8, 37)  # CDM mass grid


def dlogm(C, S, nmap, sel_c, sel_s):
    mc = np.sort(C[sel_c, 1])[::-1]; ms = np.sort(S[sel_s, 1])[::-1]
    nc = np.arange(1, len(mc) + 1) / nmap; ns = np.arange(1, len(ms) + 1) / nmap
    n_at = np.interp(Mg, mc[::-1], nc[::-1])                # cumulative abundance of CDM at mass Mg
    ok = n_at <= ns.max()
    msid = np.interp(n_at, ns, ms)                          # SIDM mass at the same abundance
    return np.where(ok, msid - Mg, np.nan)


fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True)
col = dict(zip(SIG, ("#2a9d8f", "#e9a03b", "#c0392b")))
rng = np.random.default_rng(0)
for i, (lab, fc, fs) in enumerate((("all subhaloes", lambda C: np.ones(len(C), bool), None),
                                   (f"inside {RSPLIT * 0.02:.1f} Mpc", lambda C: C[:, 2] < RSPLIT, None),
                                   (f"outside {RSPLIT * 0.02:.1f} Mpc", lambda C: C[:, 2] >= RSPLIT, None))):
    g = ax[i]
    for s in SIG:
        C, S, nm_ = cat[s, "CDM"], cat[s, "SIDM"], cat[s, "nmap"]
        d = dlogm(C, S, nm_, fc(C), fc(S))
        bs = []
        for _ in range(100):
            j = rng.integers(0, nm_, nm_); w = np.bincount(j, minlength=nm_)
            Cb = np.repeat(C, w[C[:, 0].astype(int)], 0); Sb = np.repeat(S, w[S[:, 0].astype(int)], 0)
            bs.append(dlogm(Cb, Sb, nm_, fc(Cb), fc(Sb)))
        lo, hi = np.nanpercentile(bs, [16, 84], 0)
        g.fill_between(Mg, lo, hi, color=col[s], alpha=0.25, lw=0); g.plot(Mg, d, color=col[s], lw=2, label=f"σ = {s}")
    g.axhline(0, color="0.5", lw=0.7); g.set_xlabel("log10 subhalo mass in CDM (excess inside 60 kpc) [M☉]")
    g.set_title(lab)
ax[0].set_ylabel("Δ log M = log M_SIDM − log M_CDM\n(at the same abundance)"); ax[0].legend()
fig.suptitle("BAHAMAS: change in subhalo mass vs subhalo mass (abundance matched, bands = bootstrap over clusters)")
out = f"{HERE}/figures/subhalo_dmass.png"; fig.savefig(out, dpi=100); print(out)

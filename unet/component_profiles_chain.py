"""DM, stellar, gas and total surface-density profiles, BAHAMAS test split: true SIDM vs the full hybrid chain
(DM hybrid flow runs/dm_flow_hyb -> stars runs/<stars> on DM draw d -> gas runs/<gas> on both; draw d of each is one
consistent chain realisation). Linear Sigma: DM 10^l, stars 10^l - 1e11 (soft floor), gas 10^l with floored pixels empty;
total = DM + stars + gas per map. Ring means (20 kpc) per map, median over maps; chain band = 16-84% of per-draw medians.
Ratio panels relative to true SIDM, grey bands +-5% / +-10%.
usage: python component_profiles_chain.py [--stars stars_hyb_aug] [--gas gas_hyb_hyb] [--out component_profiles_chain]
"""
import argparse
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dm_common as C, two_scale as T2
ap = argparse.ArgumentParser(); ap.add_argument("--stars", default="stars_hyb_aug"); ap.add_argument("--gas", default="gas_hyb_hyb")
ap.add_argument("--out", default="component_profiles_chain"); a = ap.parse_args()
RB = np.arange(0, 47); RK = (RB[:-1] + 0.5) * 20; LB = np.digitize(T2.R, RB) - 1; OK = (LB >= 0) & (LB < len(RK)); cnt = np.bincount(LB[OK], minlength=len(RK))
LIN = {"DM": lambda l: 10 ** l.astype(np.float64), "stars": lambda l: np.maximum(10 ** l.astype(np.float64) - 1e11, 0),
       "gas": lambda l: np.where(l > 11 + 1e-4, 10 ** l.astype(np.float64), 0)}
ring = lambda S: np.array([np.bincount(LB[OK], s[OK], minlength=len(RK)) / cnt for s in S])
S, _ = C.load(); te = S["test"]; sig = te["sigma"]; k = te["k"]
Ds = np.load(f"{C.HERE}/data/pairs_stars.npz", allow_pickle=True); Dg = np.load(f"{C.HERE}/data/pairs_gas.npz", allow_pickle=True)
truth = {"DM": te["s"], "stars": Ds["y"][k][:, 2:98, 2:98], "gas": Dg["y"][k][:, 2:98, 2:98]}
draws = {"DM": "dm_flow_hyb", "stars": a.stars, "gas": a.gas}
X = {c: np.load(f"{C.HERE}/runs/{d}/draws.npz")["draws"] for c, d in draws.items()}
nd = min(len(v) for v in X.values())
PT, PC = {}, {}
lt = {c: LIN[c](truth[c]) for c in truth}
for c in truth: PT[c] = ring(lt[c])
PT["total"] = ring(lt["DM"] + lt["stars"] + lt["gas"]); del lt
PC = {c: np.zeros((nd, len(k), len(RK))) for c in ("DM", "stars", "gas", "total")}
for d in range(nd):
    ld = {c: LIN[c](X[c][d].astype(np.float32)) for c in X}
    for c in X: PC[c][d] = ring(ld[c])
    PC["total"][d] = ring(ld["DM"] + ld["stars"] + ld["gas"])
COLS = {0.1: "#2a9d8f", 0.3: "#e9a03b", 1.0: "#d1495b"}; COMP = ("DM", "stars", "gas", "total")
fig, ax = plt.subplots(2, 4, figsize=(24, 10), sharex=True, height_ratios=[2, 1.3], constrained_layout=True)
for j, c in enumerate(COMP):
    A0, A1 = ax[0, j], ax[1, j]
    A1.axhspan(0.9, 1.1, color="0.88", lw=0, zorder=0); A1.axhspan(0.95, 1.05, color="0.7", lw=0, zorder=0); A1.axhline(1, color="0.5", lw=0.6)
    for s in C.SIG:
        m = sig == s; base = np.median(PT[c][m], 0); md = np.median(PC[c][:, m], 1); lo, mid, hi = np.percentile(md, [16, 50, 84], 0)
        A0.plot(RK, base, color=COLS[s], lw=2.6, label=f"true SIDM{s:g}"); A0.plot(RK, mid, color=COLS[s], lw=2, ls="--", label=f"chain SIDM{s:g}")
        A1.plot(RK, mid / base, color=COLS[s], lw=2.2, ls="--", label=f"SIDM{s:g}"); A1.fill_between(RK, lo / base, hi / base, color=COLS[s], alpha=0.2, lw=0)
        print(f"{c:5s} SIDM{s:g} chain/true at R=10,30,50,100,200,500 kpc: " + " ".join(f"{(mid / base)[i]:.3f}" for i in (0, 1, 2, 4, 9, 24)))
    A0.set_xscale("log"); A0.set_yscale("log"); A0.set_title(c if c != "DM" else "dark matter", fontsize=15); A0.legend(fontsize=9, ncol=2)
    A1.set_ylim(0.8, 1.2); A1.set_xlabel("R [kpc]"); A1.legend(fontsize=9)
ax[0, 0].set_ylabel("Σ [Msun pMpc⁻²], median over maps"); ax[1, 0].set_ylabel("chain / true SIDM")
fig.suptitle(f"Full hybrid chain vs true SIDM, BAHAMAS test split ({nd} draws: dm_flow_hyb → {a.stars} → {a.gas}); solid true, dashed chain", fontsize=15)
fig.savefig(f"{C.HERE}/figures/{a.out}.png", dpi=65); print(f"wrote figures/{a.out}.png")

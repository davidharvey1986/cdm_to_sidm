"""One-point transfer function: a monotone map of pixel values log Sigma_CDM -> log Sigma_SIDM, learned by matching
the training suite's pixel PDFs (quantile matching), then applied pixel by pixel to CDM maps (no position info).
  per training sigma: T_s(v) = Q_SIDM(F_CDM(v)) - v      (F, Q: pooled pixel CDF / quantile function)
  across sigma:       T(v, sigma) = h(v) sigma^p       (least squares over the training sigmas, p scanned)
so it can be applied at sigmas the training suite does not have (cross-suite). T is set to 0 below VMIN = 14.0
(pixel values below are noise-dominated), and h, p are fitted only above.
Compared with the truth and with the mass+z template converter (two_scale + mass_template), on the training suite's
held-out clusters and on all of the other suite: one-point ratio, radial delta profile, power ratio.
usage: python3 pixel_transfer.py   -> runs/pixel_transfer/{transfer.json}, figures/pixel_transfer_*.png
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import two_scale as T2

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/runs/pixel_transfer"; os.makedirs(OUT, exist_ok=True)
VG = np.linspace(12.3, 15.9, 181)                     # pixel-value grid for the transfer function
VMIN = 14.0                                           # below this the measured T is noise: fitted only above, T = 0 below
QG = np.concatenate([np.linspace(0, 0.99, 991), 1 - np.logspace(-2, -6.5, 200)])
MT = json.load(open(f"{HERE}/runs/mass_template/summary.json"))
PB = np.linspace(12.4, 15.8, 69); PM = 0.5 * (PB[1:] + PB[:-1])


def suite(name):
    S = T2.load_suite(name); D = np.load(f"{HERE}/data/{'pairs.npz' if name == 'bahamas' else 'darkskies_pairs.npz'}", allow_pickle=True)
    return S


def learn(tr):
    sg = np.unique(tr["sigma"]); Ts = []
    for s in sg:
        m = tr["sigma"] == s; c = tr["c"][m].ravel(); t = tr["s"][m].ravel()
        qc, qs = np.quantile(c, QG), np.quantile(t, QG)
        qc, idx = np.unique(qc, return_index=True)
        Ts.append(np.interp(VG, qc, qs[idx] - qc))          # T_s on the value grid (flat extrapolation)
    Ts = np.array(Ts); Ts[:, VG < VMIN] = 0.0; ps = np.linspace(0.1, 2.0, 191)
    def res(p):
        w = sg ** p; h = (w @ Ts) / (w @ w); return np.sum((Ts - np.outer(w, h)) ** 2), h
    p = float(ps[np.argmin([res(p)[0] for p in ps])]); h = res(p)[1]
    return dict(sigmas=sg.tolist(), T=Ts.tolist(), p=p, h=h.tolist())


def apply_pixel(c, sigma, T):
    return c + (sigma[:, None, None] ** T["p"]) * np.interp(c, VG, np.array(T["h"]))


def apply_template(c, sigma, z, M, tag):
    J = json.load(open(f"{HERE}/runs/two_scale_{tag}_nocnn/template.json")); C = np.array(MT[tag]["mass+z"]["C"]); p = J["template"]["p"]
    bp = C[0][None] + C[1][None] * (M[:, None] - 14.5) + C[2][None] * (z[:, None] - 0.2)
    Lc = T2.smooth(c); Sc = c - Lc
    return c + (sigma[:, None, None] ** p) * bp[:, T2.RB] + (T2.apply_small(Sc, sigma, J["small"]) - Sc)


S = {n: suite(n) for n in ("bahamas", "darkskies")}
TR = {n: learn(S[n]["train"]) for n in S}
json.dump(TR, open(f"{OUT}/transfer.json", "w"))
for n in TR: print(f"{n}: pixel transfer p = {TR[n]['p']:.3f}")

# figure 1: the learned transfer functions
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, sharey=True)
cols = {0.01: "#7fb3d5", 0.05: "#2a9d8f", 0.1: "#27ae60", 0.2: "#e9a03b", 0.3: "#d35400", 1.0: "#c0392b"}
for a, n in zip(ax, TR):
    for s, Ts in zip(TR[n]["sigmas"], TR[n]["T"]):
        a.plot(VG, Ts, color=cols[round(s, 2)], lw=1.8, label=f"σ={s:g} measured")
        a.plot(VG, (s ** TR[n]["p"]) * np.array(TR[n]["h"]), color=cols[round(s, 2)], lw=1, ls="--")
    a.axhline(0, color="0.5", lw=0.7); a.set_xlabel("CDM pixel log10 Σ"); a.set_title(f"{n}: T(v) = log Σ_SIDM − log Σ_CDM at equal pixel rank\n(dashed: h(v) σ^{TR[n]['p']:.2f})", fontsize=10)
    a.set_xlim(12.5, 15.8); a.set_ylim(-0.25, 0.08); a.legend(fontsize=7)
ax[0].set_ylabel("T [dex]")
fig.savefig(f"{HERE}/figures/pixel_transfer_functions.png", dpi=100)

# figure 2: tests
LAM = (24, 8, 4, 2.5); KI = [int(round(96 / L)) for L in LAM]
tests = [("bahamas", "bahamas", "in-suite"), ("bahamas", "darkskies", "BAHAMAS → darkskies"),
         ("darkskies", "darkskies", "in-suite"), ("darkskies", "bahamas", "darkskies → BAHAMAS")]
fig, ax = plt.subplots(4, 3, figsize=(18, 19), constrained_layout=True)
rr = np.arange(48) + 0.5; rr[0] = 0.35
for i, (trn, ten, lab) in enumerate(tests):
    te = S[ten]["test"] if trn == ten else {k: np.concatenate([S[ten]["train"][k], S[ten]["test"][k]]) for k in S[ten]["train"]}
    sg = np.unique(te["sigma"]); show = [sg[-1]] if len(sg) < 4 else [sg[1], sg[-1]]
    show = sg if len(sg) == 3 else sg[[1, 3]]
    for s in show:
        m = te["sigma"] == s; c, t = te["c"][m].astype(np.float64), te["s"][m].astype(np.float64)
        pix = apply_pixel(c, te["sigma"][m], TR[trn]); tpl = apply_template(c, te["sigma"][m], te["z"][m], te["M"][m], trn)
        hc = np.histogram(c, PB)[0] + 0.5; ok = hc > 30
        for nm, v, ls in (("true", t, "-"), ("pixel transfer", pix, "--"), ("template converter", tpl, ":")):
            cl = cols[round(float(s), 2)]
            ax[i, 0].plot(PM[ok], ((np.histogram(v, PB)[0] + 0.5) / hc)[ok], color=cl, ls=ls, lw=2 if ls == "-" else 1.6,
                          label=f"σ={s:g} {nm}")
            ax[i, 1].plot(rr, np.median(T2.rprof(v - c), 0), color=cl, ls=ls, lw=2 if ls == "-" else 1.6)
            pr = T2.pk_ratio(c, v)[KI]; ax[i, 2].plot(LAM, pr, color=cl, ls=ls, marker="o", lw=2 if ls == "-" else 1.6)
    ax[i, 0].axhline(1, color="0.5", lw=0.7); ax[i, 0].set_ylim(0.3, 1.4); ax[i, 0].set_ylabel(f"trained {trn}, tested {ten}\n({lab})", fontsize=11)
    ax[i, 0].set_xlabel("pixel log10 Σ"); ax[i, 0].set_title("one-point ratio to CDM", fontsize=10); ax[i, 0].legend(fontsize=6, ncol=2)
    ax[i, 1].axhline(0, color="0.5", lw=0.7); ax[i, 1].set_xscale("log"); ax[i, 1].set_ylim(-0.2, 0.05)
    ax[i, 1].set_xlabel("r [px]"); ax[i, 1].set_title("median radial Δ (solid true, -- pixel transfer, : template)", fontsize=10)
    ax[i, 2].set_xscale("log"); ax[i, 2].set_xlabel("wavelength [px]"); ax[i, 2].set_title("power ratio P(SIDM)/P(CDM)", fontsize=10)
fig.suptitle("one-point pixel transfer vs template converter vs truth", fontsize=15)
fig.savefig(f"{HERE}/figures/pixel_transfer_tests.png", dpi=80); print("wrote figures")

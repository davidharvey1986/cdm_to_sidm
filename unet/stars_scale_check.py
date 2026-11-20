"""Does the stars step depend on the absolute stellar scale? (transfer to a suite with a different stellar-to-halo relation)

Part 1 (truth): true BAHAMAS SIDM/CDM stellar change vs the CDM central stellar fraction f* = M*(<50 kpc)/M_DM(<50 kpc).
Part 2 (flow): rescale the input CDM stars by FAC (Sigma_* -> f Sigma_*, re-floored at 1e11) and draw from the stars flow with
  the true SIDM DM (oracle, DM unchanged); does the predicted fractional change move with f? Run on CPU (the GPU is training).
Ring means per map cached in runs/stars_scale_check.npz; figure figures/stars_scale_check.png.
usage: python stars_scale_check.py [--flow] [--model=flowA_stars_aug]   (--flow: (re)run the flow draws; else plot from the cache)
"""
import sys, time
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dm_common as C, two_scale as T2
FAC, NDRAW, A_PX = (0.5, 1.0, 2.0), 4, 0.02 ** 2
RB = np.arange(0, 47); RK = (RB[:-1] + 0.5) * 20; LB = np.digitize(T2.R, RB) - 1; OK = (LB >= 0) & (LB < len(RK)); cnt = np.bincount(LB[OK], minlength=len(RK))
lin = lambda l: np.maximum(10 ** np.asarray(l, np.float64) - 1e11, 0)
ring = lambda L: np.array([np.bincount(LB[OK], lin(l)[OK], minlength=len(RK)) / cnt for l in L])  # ring-mean Sigma per map
S, _ = C.load(); te = S["test"]; sig = te["sigma"]; D = np.load(f"{C.HERE}/data/pairs_stars.npz", allow_pickle=True)
k = te["k"]; xs = D["x"][D["xi"][k]][:, 2:98, 2:98]; ys = D["y"][k][:, 2:98, 2:98]
MODEL = next((v.split("=")[1] for v in sys.argv if v.startswith("--model=")), "flowA_stars"); TAG = MODEL.replace("flowA_stars", "")
CACHE = f"{C.HERE}/runs/stars_scale_check{TAG}.npz"

if "--flow" in sys.argv:
    import torch
    from flows import Flow, cond_vec, mark_initialized
    torch.set_num_threads(6); torch.manual_seed(7); dev = "cpu"; t0 = time.time()
    ck = torch.load(f"{C.HERE}/runs/{MODEL}/model.pt", map_location=dev, weights_only=False)
    F = Flow(**ck["arch"]).to(dev); F.load_state_dict(ck["state"]); mark_initialized(F); F.eval(); nm = ck["norm"]; ds = nm["dscale"]
    UC = float(np.arcsinh(3.5 / ds)); st = lambda v, m, s: (v - m) / s; cdm = te["c"]; ddm = te["s"] - cdm; B = 64
    out = {}
    with torch.no_grad():
        for f in FAC:
            xf = np.log10(f * lin(xs) + 1e11).astype(np.float32); Pin = ring(xf); Pout = np.zeros((NDRAW,) + Pin.shape)
            for b in range(0, len(k), B):
                sl = slice(b, min(b + B, len(k))); n = sl.stop - sl.start
                cv = cond_vec(sig[sl].astype(np.float64), te["z"][sl].astype(np.float64)).to(dev)
                ci = torch.tensor(np.stack([st(xf[sl], nm["mu"], nm["sd"]), st(cdm[sl], nm["cx_mu"][0], nm["cx_sd"][0]), st(ddm[sl], nm["cy_mu"][0], nm["cy_sd"][0])], 1), dtype=torch.float32)
                for d in range(NDRAW):
                    u = F.decode(F.sample_eps(n, dev, T=1.0, H=C.N), cv, ci, H=C.N)[:, 0].numpy()
                    Pout[d, sl] = ring(xf[sl] + np.sinh(np.clip(u, -UC, UC)) * ds)
                print(f"f={f} {sl.stop}/{len(k)} {time.time() - t0:.0f}s", flush=True)
            out[f"in_{f}"] = Pin; out[f"out_{f}"] = Pout
    np.savez(CACHE, **out); print("wrote", CACHE)

C_ = np.load(CACHE); Pc, Ps = ring(xs), ring(ys)
Pd = np.array([np.bincount(LB[OK], (10 ** l.astype(np.float64))[OK], minlength=len(RK)) / cnt for l in te["c"]])
area = cnt * A_PX; cum = lambda P, j: (P[..., :j] * area[:j]).sum(-1)  # mass within ring j
J50, J100 = 3, 5   # rings 0-2 (< 60 kpc), 0-4 (< 100 kpc)
fstar = cum(Pc, J50) / cum(Pd, J50); lf = np.log10(fstar)
dtrue50 = np.log10(cum(Ps, J50) / cum(Pc, J50)); dtrue100 = np.log10(cum(Ps, J100) / cum(Pc, J100))
COLS = {0.1: "#2a9d8f", 0.3: "#e9a03b", 1.0: "#d1495b"}; FC = {0.5: "#3a6fb0", 1.0: "k", 2.0: "#d1495b"}
fig, ax = plt.subplots(2, 3, figsize=(20, 11.5), constrained_layout=True)
# row 1: truth
edges = np.percentile(lf, [0, 25, 50, 75, 100])
for s in C.SIG:
    m = sig == s; ax[0, 0].scatter(lf[m], dtrue50[m], s=6, color=COLS[s], alpha=0.35)
    xb, yb, eb = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        q = m & (lf >= a) & (lf <= b); xb.append(np.median(lf[q])); yb.append(np.median(dtrue50[q])); eb.append(1.2533 * dtrue50[q].std() / np.sqrt(q.sum()))
    ax[0, 0].errorbar(xb, yb, eb, color=COLS[s], lw=2.4, marker="o", capsize=3, label=f"SIDM{s:g} binned median")
    sl, ic = np.polyfit(lf[m], dtrue50[m], 1); r = np.corrcoef(lf[m], dtrue50[m])[0, 1]
    print(f"truth SIDM{s:g}: d log M*(<60kpc) vs log f*: slope {sl:+.3f}, Pearson r {r:+.2f}; median d {np.median(dtrue50[m]):+.3f}")
ax[0, 0].axhline(0, color="0.6", lw=0.6); ax[0, 0].set_xlabel("log10 f*,CDM = M*/M_DM (< 60 kpc)"); ax[0, 0].set_ylabel("true log10 M*_SIDM/M*_CDM (< 60 kpc)")
ax[0, 0].set_ylim(-0.5, 0.4); ax[0, 0].legend(fontsize=10); ax[0, 0].set_title("truth: central stellar change vs stellar fraction")
for j, s in enumerate((0.3, 1.0)):
    A = ax[0, 1 + j]; m = sig == s
    for qi, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        q = m & (lf >= a) & (lf <= b); rr = np.median(Ps[q], 0) / np.median(Pc[q], 0)
        A.plot(RK, rr, lw=2.2, color=plt.cm.viridis(qi / 3), label=f"f* quartile {qi + 1} (log f* {a:.2f}–{b:.2f})")
    A.axhspan(0.9, 1.1, color="0.9", lw=0, zorder=0); A.axhspan(0.95, 1.05, color="0.75", lw=0, zorder=0); A.axhline(1, color="0.5", lw=0.6)
    A.set_xscale("log"); A.set_ylim(0.7, 1.2); A.set_xlabel("R [kpc]"); A.set_ylabel("true Σ*_SIDM / Σ*_CDM (median profiles)")
    A.set_title(f"truth, SIDM{s:g}: stellar change by CDM stellar-fraction quartile"); A.legend(fontsize=9)
# row 2: flow with rescaled input stars
for j, s in enumerate(C.SIG):
    A = ax[1, j]; m = sig == s
    for f in FAC:
        Pin, Pout = C_[f"in_{f}"], C_[f"out_{f}"]; rr = np.array([np.median(Pout[d][m], 0) / np.median(Pin[m], 0) for d in range(len(Pout))])
        A.plot(RK, rr.mean(0), color=FC[f], lw=2.2, ls="-" if f == 1 else "--", label=f"flow, input stars × {f:g}")
        A.fill_between(RK, rr.min(0), rr.max(0), color=FC[f], alpha=0.15, lw=0)
        d50 = np.median(np.log10(cum(Pout[:, m], J50) / cum(Pin[m], J50))); print(f"flow SIDM{s:g} stars x{f:g}: median d log M*(<60kpc) {d50:+.3f}")
    A.plot(RK, np.median(Ps[m], 0) / np.median(Pc[m], 0), color="0.4", lw=1.4, ls=":", label="truth (× 1)")
    A.axhspan(0.9, 1.1, color="0.9", lw=0, zorder=0); A.axhspan(0.95, 1.05, color="0.75", lw=0, zorder=0); A.axhline(1, color="0.5", lw=0.6)
    A.set_xscale("log"); A.set_ylim(0.7, 1.2); A.set_xlabel("R [kpc]"); A.set_ylabel("predicted Σ*_SIDM / input Σ*_CDM")
    A.set_title(f"flow, SIDM{s:g}: rescale input CDM stars (true SIDM DM)"); A.legend(fontsize=9)
fig.suptitle(f"{MODEL}: is the stars step scale-free? Top: true dependence on stellar fraction. Bottom: flow response to rescaled input stars", fontsize=15)
fig.savefig(f"{C.HERE}/figures/stars_scale_check{TAG}.png", dpi=65); print(f"wrote figures/stars_scale_check{TAG}.png")

"""Is the gas step scale-free? Rescale the input CDM gas (x0.5, x2; hard floor 1e11 re-applied, floored pixels empty) or the
input CDM stars (x0.5, x2; soft floor) and draw from the gas flow with the true SIDM DM and stars (oracle). A scale-free
flow predicts the same fractional gas change for every input scaling. Compare --model=flowA_gas and flowA_gas_aug.
Ring means cached in runs/gas_scale_check<tag>.npz; figure figures/gas_scale_check<tag>.png.
usage: python gas_scale_check.py --flow --model=flowA_gas_aug
"""
import sys, time
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dm_common as C, two_scale as T2
VAR = (("gas", 0.5), ("gas", 1.0), ("gas", 2.0), ("stars", 0.5), ("stars", 2.0)); NDRAW, A_PX = 4, 0.02 ** 2
RB = np.arange(0, 47); RK = (RB[:-1] + 0.5) * 20; LB = np.digitize(T2.R, RB) - 1; OK = (LB >= 0) & (LB < len(RK)); cnt = np.bincount(LB[OK], minlength=len(RK))
glin = lambda l: np.where(l > 11 + 1e-4, 10 ** np.asarray(l, np.float64), 0)
ring = lambda L: np.array([np.bincount(LB[OK], glin(l)[OK], minlength=len(RK)) / cnt for l in L])
MODEL = next((v.split("=")[1] for v in sys.argv if v.startswith("--model=")), "flowA_gas"); TAG = MODEL.replace("flowA_gas", "")
CACHE = f"{C.HERE}/runs/gas_scale_check{TAG}.npz"
S, _ = C.load(); te = S["test"]; sig = te["sigma"]; k = te["k"]

if "--flow" in sys.argv:
    import torch
    from flows import Flow, cond_vec, mark_initialized
    D = np.load(f"{C.HERE}/data/pairs_gas.npz", allow_pickle=True)
    xg = D["x"][D["xi"][k]][:, 2:98, 2:98]; xs = D["cx"][D["xi"][k], 1, 2:98, 2:98]; dst = D["cy"][k, 1, 2:98, 2:98]; yg = D["y"][k][:, 2:98, 2:98]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(7); t0 = time.time()
    ck = torch.load(f"{C.HERE}/runs/{MODEL}/model.pt", map_location=dev, weights_only=False)
    F = Flow(**ck["arch"]).to(dev); F.load_state_dict(ck["state"]); mark_initialized(F); F.eval(); nm = ck["norm"]; ds = nm["dscale"]
    OFFM = np.load(f"{C.HERE}/data/{nm['offset']}")["off_ring"][k][:, T2.RB] if nm.get("offset") else 0.0
    UC = float(np.arcsinh(1.5 / ds)); st = lambda v, m, s: (v - m) / s; cdm = te["c"]; ddm = te["s"] - cdm; B = 64
    out = {"true_in": ring(xg), "true_out": ring(yg)}
    with torch.no_grad():
        for what, f in VAR:
            g_in = np.log10(np.maximum(f * glin(xg), 1e11)).astype(np.float32) if what == "gas" else xg
            s_in = np.log10(f * np.maximum(10 ** xs.astype(np.float64) - 1e11, 0) + 1e11).astype(np.float32) if what == "stars" else xs
            Pout = np.zeros((NDRAW, len(k), len(RK)))
            for b in range(0, len(k), B):
                sl = slice(b, min(b + B, len(k))); n = sl.stop - sl.start
                cv = cond_vec(sig[sl].astype(np.float64), te["z"][sl].astype(np.float64)).to(dev)
                ci = torch.tensor(np.stack([st(g_in[sl], nm["mu"], nm["sd"]), st(cdm[sl], nm["cx_mu"][0], nm["cx_sd"][0]), st(s_in[sl], nm["cx_mu"][1], nm["cx_sd"][1]),
                                            st(ddm[sl], nm["cy_mu"][0], nm["cy_sd"][0]), st(dst[sl], nm["cy_mu"][1], nm["cy_sd"][1])], 1), dtype=torch.float32, device=dev)
                for d in range(NDRAW):
                    u = F.decode(F.sample_eps(n, dev, T=1.0, H=C.N), cv, ci, H=C.N)[:, 0].cpu().numpy()
                    Pout[d, sl] = ring(np.maximum(g_in[sl] + (OFFM[sl] if np.ndim(OFFM) else 0.0) + np.sinh(np.clip(u, -UC, UC)) * ds, 11.0))
            out[f"in_{what}_{f}"] = ring(g_in); out[f"out_{what}_{f}"] = Pout; print(f"{what} x{f} done {time.time() - t0:.0f}s", flush=True)
    np.savez(CACHE, **out); print("wrote", CACHE)

Q = np.load(CACHE); area = cnt * A_PX; cum = lambda P, j: (P[..., :j] * area[:j]).sum(-1); J = 5  # < 100 kpc
FC = {("gas", 0.5): ("#3a6fb0", "--"), ("gas", 1.0): ("k", "-"), ("gas", 2.0): ("#d1495b", "--"), ("stars", 0.5): ("#3a6fb0", ":"), ("stars", 2.0): ("#d1495b", ":")}
fig, ax = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)
for i, s in enumerate(C.SIG):
    A = ax[i]; m = sig == s
    for what, f in VAR:
        Pin, Pout = Q[f"in_{what}_{f}"], Q[f"out_{what}_{f}"]; rr = np.array([np.median(P[m], 0) / np.median(Pin[m], 0) for P in Pout])
        A.plot(RK, rr.mean(0), color=FC[what, f][0], ls=FC[what, f][1], lw=2.2, label=f"input {what} × {f:g}")
        d = np.median(np.log10(cum(Pout[:, m], J) / cum(Pin[m], J))); print(f"{MODEL} SIDM{s:g} {what} x{f:g}: median d log M_gas(<100kpc) {d:+.3f}")
    A.plot(RK, np.median(Q["true_out"][m], 0) / np.median(Q["true_in"][m], 0), color="0.4", lw=1.4, ls="-.", label="truth (× 1)")
    A.axhspan(0.9, 1.1, color="0.9", lw=0, zorder=0); A.axhspan(0.95, 1.05, color="0.75", lw=0, zorder=0); A.axhline(1, color="0.5", lw=0.6)
    A.set_xscale("log"); A.set_ylim(0.7, 1.3); A.set_xlabel("R [kpc]"); A.set_ylabel("predicted Σgas_SIDM / input Σgas_CDM"); A.legend(fontsize=9)
    A.set_title(f"SIDM{s:g}: rescaled input gas (dashed) or stars (dotted)")
fig.suptitle(f"{MODEL}: is the gas step scale-free? (true SIDM DM and stars)", fontsize=15)
fig.savefig(f"{C.HERE}/figures/gas_scale_check{TAG}.png", dpi=65); print(f"wrote figures/gas_scale_check{TAG}.png")

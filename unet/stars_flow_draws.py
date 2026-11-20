"""Draws from the stars flow (train_flow.py --data pairs_stars.npz) for the dm_common test set.
Conditioning: CDM stars, CDM DM and Delta DM = SIDM DM - CDM DM, where SIDM DM is
    --dm true : the true SIDM dark matter (oracle; isolates the stars step)
    --dm hyb  : draw d of the DM hybrid flow (runs/dm_flow_hyb/draws.npz) for stars draw d (the full chain)
usage: python stars_flow_draws.py --name flowA_stars --dm hyb [--ndraw 19] -> runs/stars_<dm>/draws.npz (log10(Sigma_star + 1e11), 96x96)
"""
import os, argparse, time
import numpy as np, torch
import dm_common as C
from flows import Flow, cond_vec, mark_initialized
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="flowA_stars"); ap.add_argument("--dm", choices=["true", "hyb"], default="hyb")
ap.add_argument("--ndraw", type=int, default=19); ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--nmax", type=int, default=None)
ap.add_argument("--tag", default="", help="output runs/stars_<dm><tag>/")
a = ap.parse_args(); dev = "mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(7); t0 = time.time()
S, _ = C.load(); te = S["test"]; D = np.load(f"{C.HERE}/data/pairs_stars.npz", allow_pickle=True)
k = te["k"][:a.nmax]; assert D["ok"][k].all()
xs = D["x"][D["xi"][k]][:, 2:98, 2:98]; ys = D["y"][k][:, 2:98, 2:98]; cdm = te["c"][:len(k)]
dmd = np.load(f"{C.HERE}/runs/dm_flow_hyb/draws.npz")["draws"][:, :len(k)] if a.dm == "hyb" else None
ck = torch.load(f"{C.HERE}/runs/{a.name}/model.pt", map_location=dev, weights_only=False)
F = Flow(**ck["arch"]).to(dev); F.load_state_dict(ck["state"]); mark_initialized(F); F.eval(); nm = ck["norm"]; ds = nm["dscale"]
UC = float(np.arcsinh(3.5 / ds)); print(f"{a.name} epoch {ck['epoch']}, dm={a.dm}, {len(k)} test pairs", flush=True)
out = np.empty((a.ndraw, len(k), C.N, C.N), np.float16); B = 64
st = lambda v, m, s: (v - m) / s
with torch.no_grad():
    for b in range(0, len(k), B):
        sl = slice(b, min(b + B, len(k))); n = sl.stop - sl.start
        cv = cond_vec(te["sigma"][sl].astype(np.float64), te["z"][sl].astype(np.float64)).to(dev)
        for d in range(a.ndraw):
            ddm = (te["s"][sl] if a.dm == "true" else dmd[d % len(dmd), sl].astype(np.float32)) - cdm[sl]
            ci = np.stack([st(xs[sl], nm["mu"], nm["sd"]), st(cdm[sl], nm["cx_mu"][0], nm["cx_sd"][0]), st(ddm, nm["cy_mu"][0], nm["cy_sd"][0])], 1)
            u = F.decode(F.sample_eps(n, dev, T=a.T, H=C.N), cv, torch.tensor(ci, dtype=torch.float32, device=dev), H=C.N)[:, 0].cpu().numpy()
            out[d, sl] = xs[sl] + np.sinh(np.clip(u, -UC, UC)) * ds
        print(f"  {sl.stop}/{len(k)}, {time.time() - t0:.0f}s", flush=True)
od = f"{C.HERE}/runs/stars_{a.dm}{a.tag}"; os.makedirs(od, exist_ok=True)
np.savez(f"{od}/draws.npz", draws=out, k=k, name=a.name, epoch=ck["epoch"], dm=a.dm); print("wrote", od + "/draws.npz")

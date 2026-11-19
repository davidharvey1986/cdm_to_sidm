"""Draws from a flow-A model trained on the DM-only pairs (train_flow.py --data pairs_dm.npz) for the dm_common test set.
usage: python3 dm_flow_draws.py --name flowA_dm [--ndraw 19] [--T 1.0] -> runs/dm_flow/draws.npz
       python3 dm_flow_draws.py --name flowA_dm_hyb --hybrid -> runs/dm_flow_hyb/draws.npz (draw = converter mean mu + flow residual)
       --split val -> draws_val.npz (for calibrate_dm_flow.py)"""
import os, argparse, time
import numpy as np, torch
import dm_common as C
from flows import Flow, cond_vec, mark_initialized
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="flowA_dm"); ap.add_argument("--ndraw", type=int, default=19)
ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--hybrid", action="store_true")
ap.add_argument("--split", default="test", choices=["train", "val", "test"]); a = ap.parse_args()
dev = "mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(5); t0 = time.time()
S, D = C.load(); te = S[a.split]
ck = torch.load(f"{C.HERE}/runs/{a.name}/model.pt", map_location=dev, weights_only=False)
F = Flow(**ck["arch"]).to(dev); F.load_state_dict(ck["state"]); mark_initialized(F); F.eval(); nm = ck["norm"]
out = np.empty((a.ndraw, len(te["k"]), C.N, C.N), np.float16); B = 64
with torch.no_grad():
    for b in range(0, len(te["k"]), B):
        c = te["c"][b:b + B]; n = len(c); base = te["mu"][b:b + B] if a.hybrid else c
        cimg = torch.tensor((c - nm["mu"]) / nm["sd"], dtype=torch.float32, device=dev)[:, None]
        cv = cond_vec(te["sigma"][b:b + B].astype(np.float64), te["z"][b:b + B].astype(np.float64)).to(dev)
        for d in range(a.ndraw):
            u = F.decode(F.sample_eps(n, dev, T=a.T, H=C.N), cv, cimg, H=C.N)[:, 0].cpu().numpy()
            out[d, b:b + n] = base + np.sinh(np.clip(u, -4, 4)) * nm["dscale"]
        print(f"  {b + n}/{len(te['k'])}, {time.time() - t0:.0f}s", flush=True)
od = f"{C.HERE}/runs/dm_flow" + ("_hyb" if a.hybrid else ""); os.makedirs(od, exist_ok=True)
fn = f"{od}/draws.npz" if a.split == "test" else f"{od}/draws_{a.split}.npz"
np.savez(fn, draws=out, k=te["k"], name=a.name, T=a.T, hybrid=a.hybrid, split=a.split); print("wrote", fn)

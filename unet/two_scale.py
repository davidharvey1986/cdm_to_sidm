"""Two-scale CDM -> SIDM converter, trained on one suite and tested on both (cross-suite transfer).

Split every log10 map m into a large-scale part L = G(m, SL) and a small-scale part S = m - L.
Below ~3 px CDM and SIDM runs share no structure (subhalo positions are independent), above ~4 px they are
the same objects, so:
  deterministic: d = sigma^p * (A(r) + G(f(CDM_L, r, z, log sigma)))
               A(r), p: radial template at full resolution (it carries the core) from the training suite's median
               delta profiles, power law in sigma; f: small UNet correction, smoothed by G (SLC = 4 px) and trained against
               the SLC-smoothed target only (finer structure is run-to-run noise), scaled by sigma^p so it extrapolates with the template
  small scale: statistical. S_sidm = S + g(q, rbin, sigma) |S|, q = the pixel's rank within its own map and radial
               bin (scale-free, so it can move between suites), g = b(q, rbin) sigma^p2 fitted on training quantiles.
               Subhaloes stay at the CDM positions: SIDM positions are independent of CDM, so any placement is an
               equally valid draw; only their number/height statistics carry the SIDM signal.
Small-scale mapping is off inside RCORE px (core = template). SIDM map = CDM + d + (S_mapped - S_cdm).
Suites: 'bahamas' (data/pairs.npz, train/test splits) and 'darkskies' (data/darkskies_pairs.npz; its 15 'val'
halos are the in-suite test set). Pairs with |re-centring shift| <= 2 px and sigma > 0, central 96x96 px.
usage: python3 two_scale.py --train bahamas|darkskies [--epochs 15] [--no-cnn] [--smoke]
outputs runs/two_scale_<train>/{model.pt, template.json, summary.json, results.png}
"""
import os, json, argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import gaussian_filter, maximum_filter

HERE = os.path.dirname(os.path.abspath(__file__))
SL, C0, C1, N = 2.0, 2, 98, 96
yy, xx = np.indices((N, N)); R = np.hypot(yy - 47.5, xx - 47.5); RB = np.minimum(R.astype(int), 47)
RBINS, QG = (0, 15, 30, 99), np.linspace(0.005, 0.995, 100); RCORE = 6.0  # inside RCORE the small-scale map is left alone: the core is deterministic (template); QG = np.linspace(0.005, 0.995, 100)
CORE = R < 2; FIELD = R < 46
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load_suite(name, smoke=False):
    f = {"bahamas": "pairs.npz", "darkskies": "darkskies_pairs.npz"}[name]
    D = np.load(f"{HERE}/data/{f}", allow_pickle=True)
    x, y, xi, sig, sh, spl, z, M = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift", "log10_M200"))
    ok = (np.abs(sh).max(1) <= 2) & (sig > 0)
    test_name = "test" if name == "bahamas" else "val"
    out = {}
    for part, lab in (("train", "train"), ("test", test_name)):
        k = np.where(ok & (spl[xi] == lab))[0]
        if smoke: k = k[:: max(1, len(k) // 300)]
        out[part] = dict(c=x[xi[k], C0:C1, C0:C1].astype(np.float32), s=y[k, C0:C1, C0:C1].astype(np.float32),
                         sigma=sig[k].astype(np.float32), z=z[xi[k]].astype(np.float32), M=M[xi[k]].astype(np.float32),
                         cid=xi[k])
    return out


def smooth(m):
    return gaussian_filter(m, (0, SL, SL), mode="reflect")


def rprof(d):
    return np.stack([np.bincount(RB.ravel(), v.ravel(), 48) for v in d]) / np.bincount(RB.ravel(), minlength=48)


# ---------------- large-scale template
def fit_template(tr):
    dL = tr["s"] - tr["c"]; sg = np.unique(tr["sigma"])
    med_p = np.stack([np.median(rprof(dL[tr["sigma"] == s]), 0) for s in sg])
    med_c = np.array([np.median((dL[tr["sigma"] == s] * CORE).sum((1, 2)) / CORE.sum()) for s in sg])
    ps = np.linspace(0.1, 1.5, 281)
    err = [np.sum((med_c - (med_c @ sg ** p) / (sg ** p @ sg ** p) * sg ** p) ** 2) for p in ps]
    p = float(ps[np.argmin(err)]); A = (med_p.T @ sg ** p) / (sg ** p @ sg ** p)
    return dict(p=p, A=A.tolist(), sigmas=sg.tolist(), med_core=med_c.tolist())


# ---------------- small-scale quantile mapping
def ranks_by_bin(S):
    """per map, per radial bin: rank (0..1) of each pixel's small-scale value"""
    q = np.empty_like(S)
    for lo, hi in zip(RBINS[:-1], RBINS[1:]):
        m = (R >= lo) & (R < hi); v = S[:, m]
        q[:, m] = (np.argsort(np.argsort(v, 1), 1) + 0.5) / v.shape[1]
    return q


def fit_small(tr):
    Sc, Ss = tr["c"] - smooth(tr["c"]), tr["s"] - smooth(tr["s"]); sg = np.unique(tr["sigma"])
    f = np.zeros((len(sg), len(RBINS) - 1, len(QG)))
    for i, s in enumerate(sg):
        m = tr["sigma"] == s
        for j, (lo, hi) in enumerate(zip(RBINS[:-1], RBINS[1:])):
            rm = (R >= lo) & (R < hi)
            qc = np.quantile(Sc[m][:, rm], QG); qs = np.quantile(Ss[m][:, rm], QG)
            f[i, j] = (qs - qc) / np.maximum(np.abs(qc), 1e-3)
    ps = np.linspace(0.1, 2.0, 191)
    def resid(p):
        w = sg ** p; b = np.tensordot(w, f, (0, 0)) / (w @ w); return np.sum((f - w[:, None, None] * b) ** 2), b
    p2 = float(ps[np.argmin([resid(p)[0] for p in ps])]); b = resid(p2)[1]
    return dict(p2=p2, b=b.tolist())


def apply_small(Sc, sigma, sm):
    q = ranks_by_bin(Sc); b = np.array(sm["b"]); out = Sc.copy()
    for j, (lo, hi) in enumerate(zip(RBINS[:-1], RBINS[1:])):
        rm = (R >= lo) & (R < hi)
        g = np.interp(q[:, rm], QG, b[j]) * (sigma[:, None] ** sm["p2"])
        out[:, rm] = Sc[:, rm] + g * np.abs(Sc[:, rm])
    out[:, R < RCORE] = Sc[:, R < RCORE]
    return out


# ---------------- large-scale CNN correction
class Net(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())
        self.e0, self.e1, self.e2 = blk(4, w), blk(w, 2 * w), blk(2 * w, 3 * w)
        self.d1, self.d0 = blk(5 * w, 2 * w), blk(3 * w, w)
        self.out = nn.Conv2d(w, 1, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h0 = self.e0(x); h1 = self.e1(F.avg_pool2d(h0, 2)); h2 = self.e2(F.avg_pool2d(h1, 2))
        u1 = self.d1(torch.cat([F.interpolate(h2, scale_factor=2), h1], 1))
        return self.out(self.d0(torch.cat([F.interpolate(u1, scale_factor=2), h0], 1)))


RIM = torch.tensor(R / 48.0, dtype=torch.float32)
SLC = 4.0  # the CNN correction lives on >~ SLC-px scales only (4-8 px structure is run-to-run noise)
_g = np.exp(-0.5 * (np.arange(-12, 13) / SLC) ** 2); _g /= _g.sum()
GK = torch.tensor(np.outer(_g, _g), dtype=torch.float32)[None, None]


def tsmooth(v):
    """Gaussian SLC smoothing of (B,1,H,W), reflect padding"""
    return F.conv2d(F.pad(v, (12, 12, 12, 12), mode="reflect"), GK.to(v.device))


def net_input(Lc, sigma, z):
    B = len(Lc)
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    return torch.stack([(t(Lc) - 14.0) / 0.7, RIM.expand(B, N, N),
                        t(z)[:, None, None].expand(B, N, N), t(np.log10(sigma))[:, None, None].expand(B, N, N)], 1)


def predict_large(Lc, sigma, z, tpl, net=None):
    base = np.array(tpl["A"])[RB][None] * 1.0
    corr = 0.0
    if net is not None:
        with torch.no_grad():
            corr = np.concatenate([tsmooth(net(net_input(Lc[i:i + 64], sigma[i:i + 64], z[i:i + 64]).to(dev)))[:, 0].cpu().numpy()
                                   for i in range(0, len(Lc), 64)])
    return (sigma[:, None, None] ** tpl["p"]) * (base + corr)


def train_net(tr, tpl, epochs, log):
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    Lc, dLt = smooth(tr["c"]), gaussian_filter(tr["s"] - tr["c"], (0, SLC, SLC), mode="reflect")
    w = torch.tensor(FIELD, dtype=torch.float32, device=dev)
    A = torch.tensor(np.array(tpl["A"])[RB], dtype=torch.float32, device=dev)
    nb = len(Lc) // 32; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * nb)
    for ep in range(epochs):
        t0 = time.time(); perm = np.random.permutation(len(Lc)); tl = []
        for b in range(nb):
            k = perm[b * 32:(b + 1) * 32]; kr, fl = np.random.randint(4), np.random.rand() < 0.5
            aug = lambda v: np.ascontiguousarray((np.rot90(v, kr, (1, 2)))[:, :, ::-1] if fl else np.rot90(v, kr, (1, 2)))
            X = net_input(aug(Lc[k]), tr["sigma"][k], tr["z"][k]).to(dev)
            T = torch.tensor(aug(dLt[k]), dtype=torch.float32, device=dev)
            sp = torch.tensor(tr["sigma"][k] ** tpl["p"], dtype=torch.float32, device=dev)[:, None, None]
            pred = tsmooth((sp * (A + tsmooth(net(X))[:, 0]))[:, None])[:, 0]  # compared at large scales only
            loss = (((pred - T) ** 2) * w).sum() / (w.sum() * len(k))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl.append(loss.item())
        log(f"  epoch {ep}: train mse {np.mean(tl):.6f}  ({time.time() - t0:.0f}s)")
    return net.eval()


# ---------------- evaluation
def peak_stats(m):
    hp = m - gaussian_filter(m, (0, 3, 3))
    pk = (maximum_filter(hp, (1, 5, 5)) == hp) & (hp > 0.15) & (R > 6)[None]
    pk[:, :6] = pk[:, -6:] = 0; pk[:, :, :6] = pk[:, :, -6:] = 0
    return pk.sum((1, 2)).mean(), (pk & (hp > 0.7)).sum((1, 2)).mean(), np.median(hp[pk])


def pk_ratio(c, s):
    w = np.hanning(N)[:, None] * np.hanning(N)[None]
    P = lambda m: np.abs(np.fft.fft2((m - m.mean((1, 2), keepdims=True)) * w)) ** 2
    kk = np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")); kb = (kk * N).astype(int).ravel()
    pr = lambda m: np.stack([np.bincount(kb, f.ravel(), N)[:N // 2] for f in P(m)])
    return np.median(pr(s) / pr(c), 0)


def evaluate(te, tpl, sm, net):
    Lc, Sc = smooth(te["c"]), te["c"] - smooth(te["c"]); dLt = smooth(te["s"]) - Lc; dt = te["s"] - te["c"]
    preds = {"template": predict_large(Lc, te["sigma"], te["z"], tpl)}
    if net is not None: preds["template+cnn"] = predict_large(Lc, te["sigma"], te["z"], tpl, net)
    Ss = apply_small(Sc, te["sigma"], sm); dLC = gaussian_filter(dt, (0, SLC, SLC), mode="reflect")
    res = {}
    for s in np.unique(te["sigma"]):
        m = te["sigma"] == s; R_ = dict(n=int(m.sum()))
        core_t = (dt[m] * CORE).sum((1, 2)) / CORE.sum(); prof_t = np.median(rprof(dt[m]), 0)
        R_["true_core_median"] = float(np.median(core_t)); R_["true_ring"] = float(prof_t[3:10].mean())
        R_["zero_core_rmse"] = float(np.sqrt(np.mean(core_t ** 2)))
        R_["zero_field_mse"] = float(np.mean(dLt[m][:, FIELD] ** 2))
        for nm, P in preds.items():
            c = (P[m] * CORE).sum((1, 2)) / CORE.sum(); pp = np.median(rprof(P[m]), 0)
            R_[nm] = dict(core_median=float(np.median(c)), core_rmse=float(np.sqrt(np.mean((c - core_t) ** 2))),
                          core_corr=float(np.corrcoef(c, core_t)[0, 1]) if c.std() > 0 else 0.0, ring=float(pp[3:10].mean()),
                          field_r2=float(1 - np.mean((gaussian_filter(P[m], (0, SLC, SLC)) - dLC[m])[:, FIELD] ** 2) / np.mean(dLC[m][:, FIELD] ** 2)),
                          prof_rmse_r10=float(np.sqrt(np.mean((pp - prof_t)[:10] ** 2))))
        P_ = preds.get("template+cnn", preds["template"])[m]
        full = te["c"][m] + P_ + (Ss[m] - Sc[m])
        nopr = te["c"][m] + P_; tonly = te["c"][m] + preds["template"][m]
        R_["peaks"] = {k_: [float(v) for v in peak_stats(v_)] for k_, v_ in (("cdm", te["c"][m]), ("true", te["s"][m]),
                                                                              ("converted", full), ("no_small_map", nopr), ("template_only", tonly))}
        LAM = (24, 8, 4, 2.5); KI = [int(round(N / L)) for L in LAM]
        R_["power_ratio"] = {k_: pk_ratio(te["c"][m], v_)[KI].round(3).tolist() for k_, v_ in (("true", te["s"][m]), ("converted", full), ("no_small_map", nopr), ("template_only", tonly))}
        res[str(float(s))] = R_
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=("bahamas", "darkskies"), required=True)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--no-cnn", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0)
    OUT = f"{HERE}/runs/two_scale_{a.train}" + ("_smoke" if a.smoke else "") + ("_nocnn" if a.no_cnn else ""); os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/log.txt", "w")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    other = "darkskies" if a.train == "bahamas" else "bahamas"
    S_tr, S_ot = load_suite(a.train, a.smoke), load_suite(other, a.smoke)
    tr = S_tr["train"]
    log(f"train on {a.train}: {len(tr['c'])} pairs, sigma {sorted(set(tr['sigma'].tolist()))}")
    tpl = fit_template(tr); sm = fit_small(tr)
    log(f"template: core = {np.array(tpl['A'])[0]:.4f} sigma^{tpl['p']:.3f} (r=0 bin); small-scale p2 = {sm['p2']:.3f}")
    json.dump(dict(template=tpl, small=sm), open(f"{OUT}/template.json", "w"))
    net = None if a.no_cnn else train_net(tr, tpl, 2 if a.smoke else a.epochs, log)
    if net is not None: torch.save(net.state_dict(), f"{OUT}/model.pt")
    summary = {f"{a.train} (in-suite test)": evaluate(S_tr["test"], tpl, sm, net), f"{other} (cross-suite, all held-out)": None}
    ot = {k: np.concatenate([S_ot["train"][k], S_ot["test"][k]]) for k in S_ot["train"]}
    summary[f"{other} (cross-suite, all held-out)"] = evaluate(ot, tpl, sm, net)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    for title, res in summary.items():
        log(f"\n=== {title}")
        for s, R_ in res.items():
            log(f" sigma {s} (n={R_['n']}): true core {R_['true_core_median']:+.3f}, ring {R_['true_ring']:+.4f}; zero core RMSE {R_['zero_core_rmse']:.4f}")
            for nm in ("template", "template+cnn"):
                if nm in R_:
                    q = R_[nm]
                    log(f"   {nm:13s} core {q['core_median']:+.3f} RMSE {q['core_rmse']:.4f} corr {q['core_corr']:+.2f} | ring {q['ring']:+.4f} | field R2 {q['field_r2']:+.3f} | profRMSE(r<10) {q['prof_rmse_r10']:.4f}")
            pk_ = R_["peaks"]
            log("   peaks/map, bright, median height: " + "  ".join(f"{k}: {v[0]:.1f}/{v[1]:.2f}/{v[2]:.3f}" for k, v in pk_.items()))
            log("   power ratio @24/8/4/2.5 px: " + "  ".join(f"{k} {v}" for k, v in R_["power_ratio"].items()))
    log("wrote " + OUT)

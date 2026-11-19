"""sigma-conditioned UNet: CDM total-mass map -> SIDM(sigma) map.

The network predicts delta = log10(Sigma_SIDM) - log10(Sigma_CDM) on the
100x100 BAHAMAS stamp grid, conditioned on (sigma, redshift) through FiLM
(per-block scale and shift from a small MLP).

Use on any CDM cluster (same quantity, units, pixel scale and 100x100 grid as
the BAHAMAS +baryons total-mass stamps):

    from sidm_unet import load, predict
    model = load("runs/<name>/model.pt")
    sidm = predict(model, cdm_map, sigma=0.5, z=0.2)   # linear map in, linear map out
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 6  # 100 -> 112 = 7 * 2^4


def cond_features(sigma, z):
    """Conditioning vector: sigma enters linearly and as sqrt so small sigmas are resolved."""
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    z = torch.as_tensor(z, dtype=torch.float32)
    return torch.stack([sigma, torch.sqrt(sigma), z], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout, ncond):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.film = nn.Linear(ncond, 2 * cout)

    def forward(self, x, h):
        x = F.silu(self.n1(self.c1(x)))
        g, b = self.film(h).chunk(2, dim=-1)
        x = self.n2(self.c2(x)) * (1 + g[..., None, None]) + b[..., None, None]
        return F.silu(x)


class SIDMUNet(nn.Module):
    def __init__(self, width=32, ncond_hidden=64):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width, 8 * width]
        self.cond = nn.Sequential(nn.Linear(3, ncond_hidden), nn.SiLU(),
                                  nn.Linear(ncond_hidden, ncond_hidden), nn.SiLU())
        h = ncond_hidden
        self.down = nn.ModuleList([FiLMBlock(1, w[0], h)] +
                                  [FiLMBlock(w[i], w[i + 1], h) for i in range(3)])
        self.mid = FiLMBlock(w[3], w[4], h)
        self.up = nn.ModuleList([FiLMBlock(w[i + 1] + w[i], w[i], h) for i in (3, 2, 1, 0)])
        self.out = nn.Conv2d(w[0], 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start at delta = 0

    def forward(self, x, cond):
        """x: (B,1,100,100) normalised log10 CDM map; cond: (B,3). Returns delta (B,1,100,100)."""
        h = self.cond(cond)
        x = F.pad(x, (PAD,) * 4, mode="reflect")
        skips = []
        for blk in self.down:
            x = blk(x, h); skips.append(x); x = F.avg_pool2d(x, 2)
        x = self.mid(x, h)
        for blk in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skips.pop()], dim=1), h)
        return self.out(x)[..., PAD:-PAD, PAD:-PAD]


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = SIDMUNet(**ck["arch"]).to(device)
    m.load_state_dict(ck["state"]); m.eval()
    m.norm = ck["norm"]
    return m


@torch.no_grad()
def predict(model, cdm, sigma, z, device=None):
    """cdm: (100,100) or (N,100,100) linear total-mass map(s). sigma, z: scalars or (N,).
    Returns the predicted SIDM map(s), same shape and units as cdm."""
    device = device or next(model.parameters()).device
    cdm = np.asarray(cdm, dtype=np.float64)
    single = cdm.ndim == 2
    lx = np.log10(cdm.reshape(-1, 100, 100))
    n = len(lx)
    lin_in = model.norm.get("input_space", "log") == "linear"
    xin = torch.tensor(((10 ** lx if lin_in else lx) - model.norm["mu"]) / model.norm["sd"],
                       dtype=torch.float32, device=device)[:, None]
    c = cond_features(np.broadcast_to(sigma, n).copy(), np.broadcast_to(z, n).copy()).to(device)
    d = model(xin, c)[:, 0].cpu().numpy().astype(np.float64)
    if model.norm.get("target", "logratio") == "ratio":
        out = cdm.reshape(-1, 100, 100) * np.maximum(1 + d, 1e-3)  # model predicts SIDM/CDM - 1
    else:
        out = 10 ** (lx + d)
    return out[0] if single else out

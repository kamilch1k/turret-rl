"""Perception stage: synthetic drone detection, the 'image' half of the pipeline.

Renders a small grayscale camera frame with a dark drone blob whose apparent size
falls off with range (correct 1/R physics), trains a tiny CNN to detect + locate
it, and measures detection rate vs range. The falloff curve is the point: it shows
where perception breaks — a small FPV at range is a handful of pixels.

This is the honest reading of "image RL": a separate detector feeding the fire-
control policy, NOT pixels trained end-to-end into the controller (which never
sees pixels in a real system, and wouldn't converge overnight).

    python detect.py            # train + plot detection-vs-range curve

ponytail: clean sky background, no clutter/distractors — isolates the range/size
effect, which is the whole point. Add distractors + false-positive metrics next.
"""

import sys

import numpy as np

IMG = 64
NEAR, FAR = 20.0, 120.0     # range sweep, metres
K_APPARENT = 120.0          # blob radius (px) = K / range  -> ~5px near, ~1px far
NOISE = 0.10                # sensor noise std; far blobs must beat this to be seen


def render_frame(rng, present=None, range_m=None):
    """Return (img[IMG,IMG] float01, label[present,u,v]).

    Two range effects, both real: apparent size ~ 1/R, and contrast fades with
    range (atmospheric haze). Far drones become ~1px at low contrast, so they sink
    into sensor noise -- which is exactly why long-range detection fails."""
    img = np.linspace(0.75, 0.55, IMG)[:, None].repeat(IMG, 1)      # sky: brighter at top
    img = img + rng.normal(0, NOISE, (IMG, IMG))
    if present is None:
        present = rng.random() > 0.3                               # 30% empty frames
    u = v = 0.0
    if present:
        R = range_m if range_m is not None else rng.uniform(NEAR, FAR)
        rad = float(np.clip(K_APPARENT / R, 0.6, 8.0))
        amp = 0.6 * float(np.clip((NEAR / R) ** 0.9, 0.12, 1.0))   # haze: contrast fades with range
        u, v = rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88)
        yy, xx = np.mgrid[0:IMG, 0:IMG]
        blob = np.exp(-((xx - u * IMG) ** 2 + (yy - v * IMG) ** 2) / (2 * rad ** 2))
        img = img - amp * blob                                     # drone is dark vs sky
    return np.clip(img, 0, 1).astype(np.float32), np.array([float(present), u, v], np.float32)


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_net():
    torch, nn = _torch()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten())
            self.head = nn.Sequential(nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 3))

        def forward(self, x):
            return self.head(self.body(x))   # [presence_logit, u, v]

    return Net()


def _batch(rng, n=128):
    torch, _ = _torch()
    xs, ys = zip(*(render_frame(rng) for _ in range(n)))
    x = torch.tensor(np.array(xs))[:, None]
    y = torch.tensor(np.array(ys))
    return x, y


def train_detector(steps=8000):
    torch, nn = _torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training detector on {dev}")
    rng = np.random.default_rng(0)
    net = make_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    for i in range(steps):
        x, y = _batch(rng)
        x, y = x.to(dev), y.to(dev)
        out = net(x)
        mask = y[:, 0] > 0.5
        loss = bce(out[:, 0], y[:, 0])
        if mask.any():
            loss = loss + 5.0 * mse(out[mask, 1:], y[mask, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 1000 == 0:
            print(f"  step {i + 1}/{steps}  loss {loss.item():.3f}")
    torch.save(net.state_dict(), "detector.pt")
    return net, dev


def eval_vs_range(net, dev, per_bin=400, tol=0.09):
    """Detection rate (present called + localized within tol) per range bin."""
    torch, _ = _torch()
    rng = np.random.default_rng(1)
    edges = np.linspace(NEAR, FAR, 11)
    mids, rates = [], []
    net.eval()
    for lo, hi in zip(edges[:-1], edges[1:]):
        ok = 0
        for _ in range(per_bin):
            R = rng.uniform(lo, hi)
            img, lab = render_frame(rng, present=True, range_m=R)
            with torch.no_grad():
                out = net(torch.tensor(img)[None, None].to(dev)).cpu().numpy()[0]
            err = np.hypot(out[1] - lab[1], out[2] - lab[2])
            if out[0] > 0 and err < tol:
                ok += 1
        mids.append((lo + hi) / 2)
        rates.append(ok / per_bin)
    return mids, rates


def main():
    net, dev = train_detector()
    mids, rates = eval_vs_range(net, dev)
    for m, r in zip(mids, rates):
        print(f"  {m:5.0f} m : {r:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mids, rates, "o-", lw=2, color="tab:purple")
    ax.set(xlabel="range to drone (m)", ylabel="detection rate",
           title="Perception breaks with range: small FPV = fewer pixels", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("detection.png", dpi=120)
    print("saved detection.png + detector.pt")


def test():
    img, lab = render_frame(np.random.default_rng(0), present=True, range_m=40)
    assert img.shape == (IMG, IMG) and img.min() >= 0 and img.max() <= 1
    assert lab[0] == 1.0 and 0 <= lab[1] <= 1
    net = make_net()
    torch, _ = _torch()
    assert net(torch.zeros(2, 1, IMG, IMG)).shape == (2, 3)
    print("detect checks passed")


if __name__ == "__main__":
    test() if (len(sys.argv) > 1 and sys.argv[1] == "test") else main()

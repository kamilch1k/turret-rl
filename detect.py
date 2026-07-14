"""Perception stage: synthetic drone detection, the 'image' half of the pipeline.

Renders a small grayscale camera frame (sky gradient, horizon/ground band, clouds,
sensor noise, dark distractors) with a drone blob whose apparent size and contrast
fall off with range (1/R + haze). Trains a tiny CNN to detect + locate the drone
and measures detection rate vs range. The falloff curve is the point: a small FPV
at range is a handful of pixels and sinks into clutter.

The camera is a pinhole slaved to the turret barrel (project/backproject), so the
same model both renders sim geometry and lets the controller back-project a
detection into a track. This is the honest reading of "image RL": a detector
feeding fire control, NOT pixels trained end-to-end (a real controller consumes
tracks, not pixels, and pixel-to-action wouldn't converge overnight).

    python detect.py            # train + plot detection-vs-range curve
    python detect.py test       # fast self-checks

ponytail: apparent size uses a lumped constant K (= drone_radius x focal_px), so
it is FOV-independent — fine for a bearing sensor; decompose it if you need true
angular size. Distractors are kept modest so the near drone still dominates.
"""

import sys

import numpy as np

IMG = 64
NEAR, FAR = 20.0, 120.0     # range sweep, metres
K_APPARENT = 120.0          # blob radius (px) = K / range  -> ~6px near, ~1px far
NOISE = 0.10                # sensor noise std; far blobs must beat this to be seen
FOV = np.radians(90.0)      # camera field of view (full angle), slaved to the barrel
F_NORM = 1.0 / (2.0 * np.tan(FOV / 2.0))   # normalized focal length


def _camera_basis(az, el):
    ce = np.cos(el)
    fwd = np.array([ce * np.cos(az), ce * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    n = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else right / n  # degenerate: looking straight up
    up = np.cross(right, fwd)
    return fwd, right, up


def project(rel, az, el):
    """Pinhole projection of a turret-relative point into the barrel-slaved camera.
    Returns (in_view, u, v, R): u,v normalized [0,1], R range in metres."""
    fwd, right, up = _camera_basis(az, el)
    R = float(np.linalg.norm(rel))
    z = float(np.dot(rel, fwd))
    if z <= 1.0:
        return False, 0.0, 0.0, R           # behind camera / too close
    u = 0.5 + (float(np.dot(rel, right)) / z) * F_NORM
    v = 0.5 - (float(np.dot(rel, up)) / z) * F_NORM
    return (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0), u, v, R


def backproject(u, v, R, az, el):
    """Inverse of project: pixel (u,v) + range R -> world position estimate.
    Exact inverse when (u,v,R) are the true projection (see test)."""
    fwd, right, up = _camera_basis(az, el)
    d = fwd + ((u - 0.5) / F_NORM) * right + (-(v - 0.5) / F_NORM) * up
    return R * d / (np.linalg.norm(d) + 1e-9)


def render_frame(rng, present=None, range_m=None, uv=None, clutter=True):
    """Return (img[IMG,IMG] float01, label[present,u,v]).

    Domain-randomized background (sky gradient, ground band, clouds, noise) so a
    CNN trained here generalizes; the drone is a dark blob with size ~1/R and
    contrast fading with range (haze). Optional dark distractors add false-positive
    pressure — at far range the ~1px drone aliases with them, which is why far
    detection collapses."""
    yy, xx = np.mgrid[0:IMG, 0:IMG]

    # sky: randomized vertical gradient
    top, bot = rng.uniform(0.6, 0.85), rng.uniform(0.4, 0.65)
    img = np.linspace(top, bot, IMG)[:, None].repeat(IMG, 1)

    # ground band (random horizon height + texture)
    if rng.random() < 0.7:
        h = int(rng.uniform(0.6, 0.95) * IMG)
        img[h:, :] = rng.uniform(0.3, 0.6) + rng.normal(0, 0.05, (IMG - h, IMG))

    # clouds: a few bright soft blobs (opposite polarity to the drone)
    for _ in range(int(rng.integers(0, 4))):
        cu, cv, crad = rng.uniform(0, 1) * IMG, rng.uniform(0, 0.6) * IMG, rng.uniform(6, 16)
        img = img + rng.uniform(0.05, 0.15) * np.exp(-((xx - cu) ** 2 + (yy - cv) ** 2) / (2 * crad ** 2))

    img = img + rng.normal(0, NOISE, (IMG, IMG))

    if present is None:
        present = rng.random() > 0.3          # 30% empty frames
    u = v = 0.0
    if present:
        u, v = uv if uv is not None else (rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88))
        R = range_m if range_m is not None else rng.uniform(NEAR, FAR)
        rad = float(np.clip(K_APPARENT / R, 0.6, 8.0))
        amp = 0.6 * float(np.clip((NEAR / R) ** 0.9, 0.12, 1.0))   # haze: contrast fades with range
        img = img - amp * np.exp(-((xx - u * IMG) ** 2 + (yy - v * IMG) ** 2) / (2 * rad ** 2))

    # dark distractors (birds/debris): modest, so the near drone still dominates
    if clutter:
        for _ in range(int(rng.integers(0, 3))):
            du, dv, drad = rng.uniform(0, 1) * IMG, rng.uniform(0, 1) * IMG, rng.uniform(0.8, 2.2)
            img = img - rng.uniform(0.1, 0.28) * np.exp(-((xx - du) ** 2 + (yy - dv) ** 2) / (2 * drad ** 2))

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
    return torch.tensor(np.array(xs))[:, None], torch.tensor(np.array(ys))


def train_detector(steps=12000):
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
        if (i + 1) % 2000 == 0:
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
            if out[0] > 0 and np.hypot(out[1] - lab[1], out[2] - lab[2]) < tol:
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
    rng = np.random.default_rng(0)

    # frame contract
    img, lab = render_frame(rng, present=True, range_m=40)
    assert img.shape == (IMG, IMG) and 0 <= img.min() and img.max() <= 1
    assert lab[0] == 1.0 and 0 <= lab[1] <= 1

    # camera round-trip: backproject(project(p)) == p for an in-view point
    az, el = 0.3, 0.2
    fwd, right, up = _camera_basis(az, el)
    p = 80 * fwd + 8 * right - 5 * up            # guaranteed in front, near axis
    in_view, u, v, R = project(p, az, el)
    assert in_view, "test point should be in view"
    rec = backproject(u, v, R, az, el)
    assert np.linalg.norm(rec - p) < 1e-3, f"camera round-trip off by {np.linalg.norm(rec - p)}"

    # a point behind the camera is not in view
    assert not project(-p, az, el)[0]

    net = make_net()
    torch, _ = _torch()
    assert net(torch.zeros(2, 1, IMG, IMG)).shape == (2, 3)
    print("detect checks passed")


if __name__ == "__main__":
    test() if (len(sys.argv) > 1 and sys.argv[1] == "test") else main()

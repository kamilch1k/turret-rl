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


def _background(rng, img=IMG):
    """Domain-randomized sky/ground/cloud/noise background (no drone). Returns (im, xx, yy)."""
    yy, xx = np.mgrid[0:img, 0:img]
    top, bot = rng.uniform(0.6, 0.85), rng.uniform(0.4, 0.65)     # sky gradient
    im = np.linspace(top, bot, img)[:, None].repeat(img, 1)
    if rng.random() < 0.7:                                        # ground band
        h = int(rng.uniform(0.6, 0.95) * img)
        im[h:, :] = rng.uniform(0.3, 0.6) + rng.normal(0, 0.05, (img - h, img))
    for _ in range(int(rng.integers(0, 4))):                      # clouds (bright, opposite polarity)
        cu, cv, crad = rng.uniform(0, 1) * img, rng.uniform(0, 0.6) * img, rng.uniform(6, 16)
        im = im + rng.uniform(0.05, 0.15) * np.exp(-((xx - cu) ** 2 + (yy - cv) ** 2) / (2 * crad ** 2))
    return im + rng.normal(0, NOISE, (img, img)), xx, yy


def _distractors(im, rng, xx, yy):
    """Dark birds/debris — false-positive pressure, kept modest so the near drone dominates."""
    h, w = im.shape
    for _ in range(int(rng.integers(0, 3))):
        du, dv, drad = rng.uniform(0, 1) * w, rng.uniform(0, 1) * h, rng.uniform(0.8, 2.2)
        im = im - rng.uniform(0.1, 0.28) * np.exp(-((xx - du) ** 2 + (yy - dv) ** 2) / (2 * drad ** 2))
    return im


def render_frame(rng, present=None, range_m=None, uv=None, clutter=True):
    """Blob drone (fast, FOV-independent size ~1/R) on a randomized background.
    Returns (img[IMG,IMG] float01, label[present,u,v])."""
    im, xx, yy = _background(rng)
    if present is None:
        present = rng.random() > 0.3          # 30% empty frames
    u = v = 0.0
    if present:
        u, v = uv if uv is not None else (rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88))
        R = range_m if range_m is not None else rng.uniform(NEAR, FAR)
        rad = float(np.clip(K_APPARENT / R, 0.6, 8.0))
        amp = 0.6 * float(np.clip((NEAR / R) ** 0.9, 0.12, 1.0))   # haze: contrast fades with range
        im = im - amp * np.exp(-((xx - u * IMG) ** 2 + (yy - v * IMG) ** 2) / (2 * rad ** 2))
    if clutter:
        im = _distractors(im, rng, xx, yy)
    return np.clip(im, 0, 1).astype(np.float32), np.array([float(present), u, v], np.float32)


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
                nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten())  # resolution-agnostic (64 or 96px); no-op at 64
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


# --- realistic 3D drone path (moderngl renderer) ---
IMG3D, NEAR3D, FAR3D, SIZE3D = 96, 12.0, 84.0, 4.0


def _make_renderer():
    import render3d
    return render3d.DroneRenderer(img=IMG3D, fov_deg=float(np.degrees(FOV)), size=SIZE3D)


def render_frame_3d(rng, renderer, present=None, range_m=None):
    """Lit 3D quad (moderngl) composited on the randomized background. Label from project()."""
    im, xx, yy = _background(rng, IMG3D)
    u = v = 0.0
    if present is None:
        present = rng.random() > 0.3
    if present:
        R = range_m if range_m is not None else rng.uniform(NEAR3D, FAR3D)
        az, el = rng.uniform(-np.pi, np.pi), rng.uniform(-0.1, 0.6)
        u, v = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        rel = backproject(u, v, R, az, el)                       # place drone so it lands at (u,v)
        gray, mask = renderer.render(rel, az, el)
        m = mask * float(np.clip((NEAR3D / R) ** 0.5, 0.3, 1.0))  # atmospheric contrast fade
        im = im * (1 - m) + gray * m
    im = _distractors(im, rng, xx, yy)
    return np.clip(im, 0, 1).astype(np.float32), np.array([float(present), u, v], np.float32)


def build_3d_dataset(renderer, n=16000, seed=0):
    torch, _ = _torch()
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(n):
        img, lab = render_frame_3d(rng, renderer)
        xs.append(img); ys.append(lab)
        if (i + 1) % 4000 == 0:
            print(f"  rendered {i + 1}/{n} frames")
    return torch.tensor(np.array(xs))[:, None], torch.tensor(np.array(ys))


def train_on_dataset(X, Y, epochs=25, bs=128):
    torch, nn = _torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training 3D detector on {dev} ({len(X)} frames x {epochs} epochs)")
    net = make_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, y = X[idx].to(dev), Y[idx].to(dev)
            out = net(x)
            mask = y[:, 0] > 0.5
            loss = bce(out[:, 0], y[:, 0])
            if mask.any():
                loss = loss + 5.0 * mse(out[mask, 1:], y[mask, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1}/{epochs}  loss {tot / max(1, n // bs):.3f}")
    torch.save(net.state_dict(), "detector3d.pt")
    return net, dev


def eval_vs_range_3d(net, dev, renderer, per_bin=200, tol=0.09):
    torch, _ = _torch()
    rng = np.random.default_rng(1)
    edges = np.linspace(NEAR3D, FAR3D, 11)
    mids, rates = [], []
    net.eval()
    for lo, hi in zip(edges[:-1], edges[1:]):
        ok = 0
        for _ in range(per_bin):
            R = rng.uniform(lo, hi)
            img, lab = render_frame_3d(rng, renderer, present=True, range_m=R)
            with torch.no_grad():
                out = net(torch.tensor(img)[None, None].to(dev)).cpu().numpy()[0]
            if out[0] > 0 and np.hypot(out[1] - lab[1], out[2] - lab[2]) < tol:
                ok += 1
        mids.append((lo + hi) / 2); rates.append(ok / per_bin)
    return mids, rates


def _save_montage(renderer, path="sample3d.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(7)
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
    for ax, R in zip(axs.ravel(), [15, 20, 30, 45, 15, 25, 35, 55]):
        img, _ = render_frame_3d(rng, renderer, present=True, range_m=float(R))
        ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.set_title(f"{R} m", fontsize=9); ax.axis("off")
    fig.suptitle("Rendered 3D drone frames (moderngl) — detector training images")
    fig.tight_layout(); fig.savefig(path, dpi=110); print(f"saved {path}")


def main3d():
    renderer = _make_renderer()
    _save_montage(renderer)
    X, Y = build_3d_dataset(renderer)
    net, dev = train_on_dataset(X, Y)
    mids, rates = eval_vs_range_3d(net, dev, renderer)
    for m, r in zip(mids, rates):
        print(f"  {m:5.0f} m : {r:.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mids, rates, "o-", lw=2, color="tab:green")
    ax.set(xlabel="range to drone (m)", ylabel="detection rate",
           title="Detection vs range — rendered 3D drone (moderngl)", ylim=(0, 1.02))
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("detection3d.png", dpi=120)
    print("saved detection3d.png + detector3d.pt")


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
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "test":
        test()
    elif cmd == "3d":
        main3d()
    else:
        main()

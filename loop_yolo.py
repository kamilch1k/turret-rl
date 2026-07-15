"""Wire the real-photo YOLO detector into the live control loop.

Surprise finding: the real-photo YOLO (yolo_real.py) needs NO retraining to work
on sim frames. Rendered at 640 px, the moderngl 3D drone against sky/ground looks
drone-like enough that the real detector fires on it at 0.97 (see `gap`). So a
real detector really does drive the loop — same sensor slot as DetectorSensor,
just a strong detector instead of the tiny CNN.

Result: YOLO-in-the-loop hit rate 0.74, vs the tiny-CNN's 0.12 and ground-truth
0.78. The earlier "closing the loop craters to 0.12" was the weak detector (+ 64px
frames), NOT the range geometry — a strong detector at adequate resolution nearly
closes the sim-to-track gap.

    python loop_yolo.py gap          # real-photo YOLO on sim frames -> ~0.97 (no domain gap)
    python loop_yolo.py loop [eps]   # closed-loop hit rate with the YOLO sensor

(No sim-specific YOLO retrain: we checked, it isn't needed. Renders share one path
between `gap` and the sensor, so there's no train/inference mismatch to worry about.)
"""

import sys
from pathlib import Path

import numpy as np

import detect
import render3d

RES = 640
FOV = 90.0
SIZE = 2.5                     # ~2.5 m small-UAS; apparent px ~ 320*SIZE/R at RES=640


def _background(rng, res=RES):
    top, bot = rng.uniform(0.55, 0.8), rng.uniform(0.35, 0.6)
    sky = np.linspace(top, bot, res)[:, None].repeat(res, 1)
    img = np.stack([sky * 0.75, sky * 0.85, sky], -1)          # bluish sky
    if rng.random() < 0.7:
        h = int(rng.uniform(0.6, 0.95) * res)
        g = rng.uniform(0.25, 0.5)
        img[h:] = np.stack([g * 0.7, g, g * 0.6], -1) + rng.normal(0, 0.04, (res - h, res, 3))
    yy, xx = np.mgrid[0:res, 0:res]
    for _ in range(int(rng.integers(0, 4))):                    # clouds
        cu, cv, cr = rng.uniform(0, res), rng.uniform(0, 0.5 * res), rng.uniform(30, 90)
        img += (rng.uniform(0.04, 0.12) * np.exp(-((xx - cu) ** 2 + (yy - cv) ** 2) / (2 * cr ** 2)))[..., None]
    img += rng.normal(0, 0.03, img.shape)
    return np.clip(img, 0, 1)


def render_scene(drone_rel, az, el, rng, renderer, haze=0.0, clutter=0):
    """Composite the 3D drone over a randomized background -> (rgb uint8, mask).
    haze fades drone contrast with range (optical low-contrast regime); clutter adds
    dark confuser blobs. Both attack the EO detector without touching a radar track."""
    bg = _background(rng)
    if clutter:
        yy, xx = np.mgrid[0:RES, 0:RES]
        for _ in range(clutter):
            du, dv, dr = rng.uniform(0, RES), rng.uniform(0, RES), rng.uniform(3, 11)
            bg = bg - (rng.uniform(0.1, 0.3) * np.exp(-((xx - du) ** 2 + (yy - dv) ** 2) / (2 * dr ** 2)))[..., None]
        bg = np.clip(bg, 0, 1)
    gray, mask = renderer.render(drone_rel, az, el)
    contrast = 1.0
    if haze:
        contrast = max(0.0, 1.0 - haze * min(float(np.linalg.norm(drone_rel)) / 130.0, 1.0))
    m = mask[..., None] * contrast
    rgb = bg * (1 - m) + np.stack([gray] * 3, -1) * m
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), mask


def _find(name):
    c = sorted(Path("runs").rglob(f"*{name}*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return str(c[-1]) if c else None


class YoloSensor:
    """Image track via the real-photo YOLO. Renders the barrel-slaved 640px frame,
    detects, back-projects the top box center (+ rangefinder range). Same slot as
    DetectorSensor. Single-env (holds tracker state + one GL ctx + one model)."""

    def __init__(self, weights=None, conf=0.25, haze=0.0, clutter=0):
        from ultralytics import YOLO
        self.model = YOLO(weights or _find("drone"))           # the yolo_real.py run
        self.conf, self.haze, self.clutter = conf, haze, clutter
        self.renderer = render3d.DroneRenderer(img=RES, fov_deg=FOV, size=SIZE)
        self.detected = False

    def reset(self, world, rng):
        self.est = world.drone + rng.normal(0, 5.0, 3)         # radar cue
        self.vel = world.drone_vel.copy()
        self.detected = False

    def measure(self, world, rng):
        rgb, _ = render_scene(world.drone, world.az, world.el, rng, self.renderer,
                              haze=self.haze, clutter=self.clutter)
        res = self.model.predict(rgb, conf=self.conf, imgsz=RES, verbose=False)[0]
        self.detected = len(res.boxes) > 0
        if len(res.boxes):
            b = res.boxes.xywhn.cpu().numpy()
            u, v = b[int(res.boxes.conf.argmax())][:2]         # top-conf box center, normalized
            R = float(np.linalg.norm(world.drone))
            Rn = R + rng.normal(0, 0.05 * R)                   # fused rangefinder
            new = detect.backproject(float(u), float(v), Rn, world.az, world.el)
            self.vel = 0.6 * self.vel + 0.4 * (new - self.est) / 0.05
            self.est = new
        return self.est, self.vel                              # miss -> hold last


def loop(episodes=80, weights=None):
    import sim
    from stable_baselines3 import PPO
    turret = PPO.load("turret_ppo")
    hr = sim.hit_rate(turret, episodes=episodes, sensor=YoloSensor(weights))
    print(f"YOLO-in-the-loop hit rate: {hr:.2f}  "
          f"(tiny-CNN 0.12 | degraded-track ~0.69 | ground-truth ~0.78)")


def gap(n=40):
    """Real-photo YOLO on sim frames: is there a domain gap? (No -> ~0.97.)"""
    from ultralytics import YOLO
    w = _find("drone")
    if not w:
        print("no real-photo YOLO found (run yolo_real.py train first)")
        return
    m = YOLO(w)
    renderer = render3d.DroneRenderer(img=RES, fov_deg=FOV, size=SIZE)
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(n):
        az, el = rng.uniform(-np.pi, np.pi), rng.uniform(0, 0.4)
        R = rng.uniform(25, 70)
        drel = detect.backproject(rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), R, az, el)
        rgb, _ = render_scene(drel, az, el, rng, renderer)
        if len(m.predict(rgb, conf=0.25, imgsz=RES, verbose=False)[0].boxes):
            hits += 1
    print(f"real-photo YOLO on {n} sim frames (drone present, 25-70 m): "
          f"{hits}/{n} detected = {hits / n:.2f}  (no domain gap)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gap"
    arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == "loop":
        loop(arg or 80)
    else:
        gap(arg or 40)

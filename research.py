"""Research: sensor fusion beats either sensor alone across regimes.

The YOLO-in-the-loop result (0.74) used clean frames. Real EO fails in haze /
clutter / at range. This hardens the EO frames until the YOLO sensor breaks, and
shows that layering a coarse always-on radar cue under the precise-but-fragile EO
detector stays robust — the textbook counter-UAS answer.

Three sensors, swept over EO difficulty (haze + clutter):
  - EO-alone  : YOLO on the rendered frame. Precise up close, collapses as it hazes.
  - radar-alone: a coarse all-range track (sim.Sensor). Image-independent -> flat.
  - layered   : EO when it detects, else radar. Should track the better of the two.

    python research.py            # run the sweep -> research.png + table

ponytail: fusion is an EO-preferred switch, not a Kalman filter — enough to show
the robustness result; add a real filter if you need calibrated covariance.
"""

import numpy as np

import sim
import loop_yolo

# radar-like coarse track: nearly always available (unlike EO), but noisy
RADAR = dict(latency=2, p_drop=0.1, noise0=6.0, ref_range=150.0)
LEVELS = [("easy", 0.0, 0), ("med", 0.5, 4), ("hard", 0.9, 8)]
EPS = 60


class LayeredSensor:
    """Radar cue fused with EO: use the precise EO track when it detects, else fall
    back to the coarse radar track (instead of holding a stale EO estimate)."""

    def __init__(self, haze=0.0, clutter=0):
        self.radar = sim.Sensor(**RADAR)
        self.eo = loop_yolo.YoloSensor(haze=haze, clutter=clutter)

    def reset(self, world, rng):
        self.radar.reset(world, rng)
        self.eo.reset(world, rng)

    def measure(self, world, rng):
        rp, rv = self.radar.measure(world, rng)                # advance both each tick
        ep, ev = self.eo.measure(world, rng)
        return (ep, ev) if self.eo.detected else (rp, rv)


def run():
    from stable_baselines3 import PPO
    turret = PPO.load("turret_ppo")

    eo, lay = [], []
    for name, haze, clut in LEVELS:
        e = sim.hit_rate(turret, episodes=EPS, sensor=loop_yolo.YoloSensor(haze=haze, clutter=clut))
        l = sim.hit_rate(turret, episodes=EPS, sensor=LayeredSensor(haze=haze, clutter=clut))
        eo.append(e); lay.append(l)
        print(f"{name:>5}: EO-alone {e:.2f} | layered {l:.2f}")
    radar = sim.hit_rate(turret, episodes=EPS, sensor=sim.Sensor(**RADAR))   # image-independent -> one value
    print(f"radar-alone (all levels): {radar:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = range(len(LEVELS))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, eo, "o-", lw=2, color="tab:blue", label="EO-alone (YOLO)")
    ax.plot(x, lay, "s-", lw=2, color="tab:green", label="layered (radar + EO)")
    ax.axhline(radar, ls="--", color="tab:orange", label=f"radar-alone ({radar:.2f})")
    for xi, (e, l) in enumerate(zip(eo, lay)):
        ax.annotate(f"{e:.2f}", (xi, e), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
        ax.annotate(f"{l:.2f}", (xi, l), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n}\nhaze={h}, clutter={c}" for n, h, c in LEVELS])
    ax.set(ylabel="closed-loop interception rate", ylim=(0, 0.85),
           title="Sensor fusion is robust: EO collapses in haze, layering holds")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("research.png", dpi=120)
    print("saved research.png")


if __name__ == "__main__":
    run()

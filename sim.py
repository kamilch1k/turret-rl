"""Counter-FPV turret sim: PPO turret vs evasive drone, with RL self-play.

A drone spawns 120-180 m out and attacks the asset at the origin. The turret
(at the origin) slews in az/el and fires ballistic projectiles with real flight
time, so it must learn lead prediction. The drone is either a scripted jinker or
its own PPO policy; `selfplay` alternately trains each side against the frozen
other and plots the resulting arms race.

Usage:
    python sim.py test              # smoke checks (physics + both env contracts)
    python sim.py train [steps]     # train turret vs scripted drone -> turret_ppo.zip
    python sim.py eval              # turret hit rate over 100 episodes
    python sim.py watch             # render an episode -> episode.gif
    python sim.py selfplay [rounds] # alternating self-play -> selfplay.png + models
"""

import sys
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

DT = 0.05                 # s per step
EP_LEN = 600              # 30 s episode
SPAWN_R = (120.0, 180.0)  # drone spawn distance, m
ASSET_R = 5.0             # drone wins inside this radius of origin
DRONE_SPEED = 25.0        # m/s cruise toward asset
DRONE_ACC = 30.0          # m/s^2 maneuver authority
MUZZLE_V = 400.0          # m/s
SLEW = 2.5                # rad/s max turret rate
COOLDOWN = 0.5            # s between shots
AMMO = 40
HIT_R = 2.0               # ponytail: proximity kill stands in for shot spread; model a pellet cone if fidelity matters
OBS_NOISE = 1.5           # m std on measured drone position
GRAV = np.array([0.0, 0.0, -9.81])


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _seg_dist(a, b, p):
    """Distance from point p to segment a-b (projectile sweep vs drone)."""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - p))


class World:
    """Shared physics. step() takes a turret action and a drone acceleration;
    both envs wrap this and translate their agent's action / build their obs."""

    def reset(self, rng):
        az = rng.uniform(-np.pi, np.pi)
        r = rng.uniform(*SPAWN_R)
        self.drone = np.array([r * np.cos(az), r * np.sin(az), rng.uniform(10, 60)])
        self.drone_vel = DRONE_SPEED * _unit(-self.drone)
        # scripted jink: two perpendicular sinusoids, random amp/freq/phase
        self.jink = rng.uniform([0.3 * DRONE_ACC, 0.5, 0], [DRONE_ACC, 2.0, 2 * np.pi], (2, 3))
        self.az, self.el = rng.uniform(-np.pi, np.pi), 0.0
        self.cool, self.ammo, self.t = 0.0, AMMO, 0
        self.shots = []  # live projectiles: [pos, vel, min_dist_to_drone]

    def barrel(self):
        ce = np.cos(self.el)
        return np.array([ce * np.cos(self.az), ce * np.sin(self.az), np.sin(self.el)])

    def scripted_accel(self):
        """Homing + perpendicular jink — the baseline (non-RL) drone."""
        vhat = _unit(self.drone_vel)
        b1 = _unit(np.cross(vhat, [0.0, 0.0, 1.0]))
        b2 = np.cross(vhat, b1)
        tt = self.t * DT
        acc = 2.0 * (DRONE_SPEED * _unit(-self.drone) - self.drone_vel)
        for (amp, freq, ph), b in zip(self.jink, (b1, b2)):
            acc = acc + amp * np.sin(freq * tt + ph) * b
        return acc

    def turret_obs(self, rng, sensor=None):
        if sensor is None:                               # clean: perfect detection, light noise
            pos = self.drone + rng.normal(0, OBS_NOISE, 3)
            vel = self.drone_vel + rng.normal(0, 1.0, 3)
        else:                                            # sensor model (degraded track or image detector)
            pos, vel = sensor.measure(self, rng)
        return np.concatenate([
            pos / 200.0, vel / 50.0,
            [np.sin(self.az), np.cos(self.az), self.el,
             self.cool / COOLDOWN, self.ammo / AMMO],
        ]).astype(np.float32)

    def drone_obs(self):
        # own state + barrel bearing + nearest incoming projectile (for dodging)
        rel_p = np.zeros(3)
        rel_v = np.zeros(3)
        has = 0.0
        if self.shots:
            near = min(self.shots, key=lambda s: np.linalg.norm(s[0] - self.drone))
            rel_p, rel_v, has = near[0] - self.drone, near[1], 1.0
        return np.concatenate([
            self.drone / 200.0, self.drone_vel / 50.0, self.barrel(),
            rel_p / 200.0, rel_v / 400.0, [has],
        ]).astype(np.float32)

    def step(self, turret_action, drone_accel):
        """Advance one tick. Returns (hit, reached, fired, nearmiss_bonus)."""
        self.t += 1

        # drone integrate (accel from script or policy), capped, floored
        self.drone_vel = self.drone_vel + drone_accel * DT
        speed = np.linalg.norm(self.drone_vel)
        if speed > 1.2 * DRONE_SPEED:
            self.drone_vel *= 1.2 * DRONE_SPEED / speed
        self.drone = self.drone + self.drone_vel * DT
        self.drone[2] = max(self.drone[2], 1.0)

        # turret slew + fire
        self.az += float(np.clip(turret_action[0], -1, 1)) * SLEW * DT
        self.el = float(np.clip(self.el + float(np.clip(turret_action[1], -1, 1)) * SLEW * DT,
                                -0.2, np.pi / 2))
        self.cool = max(0.0, self.cool - DT)
        fired = False
        if turret_action[2] > 0 and self.cool == 0.0 and self.ammo > 0:
            self.shots.append([np.zeros(3), MUZZLE_V * self.barrel(), np.inf])
            self.cool, self.ammo, fired = COOLDOWN, self.ammo - 1, True

        # projectiles: swept hit check against drone
        hit, nearmiss, live = False, 0.0, []
        for pos, vel, min_d in self.shots:
            new_pos = pos + vel * DT
            min_d = min(min_d, _seg_dist(pos, new_pos, self.drone))
            if min_d < HIT_R:
                hit = True
                continue
            vel = vel + GRAV * DT
            if new_pos[2] > 0 and np.linalg.norm(new_pos) < 500:
                live.append([new_pos, vel, min_d])
            else:
                # near-miss credit: shots scored by closest approach, else "fire" never leaves the do-nothing optimum
                nearmiss += np.exp(-min_d / 10.0)
        self.shots = live
        reached = np.linalg.norm(self.drone) < ASSET_R
        return hit, reached, fired, nearmiss


class Sensor:
    """Realistic track: detection dropout, pipeline latency, range-scaled noise.
    Real fire control never sees the true state — it sees this, and the dropout/
    latency (not the Gaussian jitter) are what actually break a perfect-info policy."""

    # defaults model a rough but defensible small-FPV track: ~300 ms pipeline latency,
    # heavy dropout that worsens with range, localization noise that grows with range
    def __init__(self, latency=6, p_drop=0.4, noise0=3.0, ref_range=150.0):
        self.latency, self.p_drop = latency, p_drop
        self.noise0, self.ref = noise0, ref_range

    def reset(self, world, rng):
        pos, vel = world.drone, world.drone_vel
        self.buf = deque([(pos.copy(), vel.copy())] * (self.latency + 1),
                         maxlen=self.latency + 1)
        self.last = (pos.copy(), vel.copy())

    def measure(self, world, rng):
        self.buf.append((world.drone.copy(), world.drone_vel.copy()))
        dpos, dvel = self.buf[0]                          # delayed by `latency` ticks
        k = np.linalg.norm(dpos) / self.ref              # range factor: worse far away
        if rng.random() < min(0.9, self.p_drop * (1 + k)):
            return self.last                             # dropout -> hold last track
        n = self.noise0 * (1 + k)
        m = (dpos + rng.normal(0, n, 3), dvel + rng.normal(0, 0.7 * n, 3))
        self.last = m
        return m


class DetectorSensor:
    """Image-based track: render the barrel-slaved camera frame, run the CNN
    detector, back-project its bearing (+ rangefinder range) to a 3D position,
    finite-difference velocity. Radar-cued at reset; holds the last track on a
    miss. Plugs into the same env slot as Sensor (reset/measure -> pos, vel).
    ponytail: single-env only (holds tracker state); use a factory for vec-envs."""

    def __init__(self, path="detector.pt"):
        import torch
        import detect
        self.torch, self.detect = torch, detect
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = detect.make_net().to(self.dev)
        self.net.load_state_dict(torch.load(path, map_location=self.dev))
        self.net.eval()

    def reset(self, world, rng):
        self.est = world.drone + rng.normal(0, 5.0, 3)   # radar cue: coarse, camera refines
        self.vel = world.drone_vel.copy()

    def measure(self, world, rng):
        det = self.detect
        in_view, u, v, R = det.project(world.drone, world.az, world.el)
        img, _ = det.render_frame(rng, present=in_view, range_m=R,
                                  uv=(u, v) if in_view else None)
        with self.torch.no_grad():
            out = self.net(self.torch.tensor(img)[None, None].to(self.dev)).cpu().numpy()[0]
        if in_view and out[0] > 0:                        # detected
            Rn = R + rng.normal(0, 0.05 * R)             # rangefinder range (monocular can't range -> fuse)
            new = det.backproject(out[1], out[2], Rn, world.az, world.el)
            self.vel = 0.6 * self.vel + 0.4 * (new - self.est) / DT   # ponytail: EMA, a real tracker filters
            self.est = new
        return self.est, self.vel                         # miss -> hold last track


class TurretEnv(gym.Env):
    """Agent = turret. Opponent drone: scripted (None) or a frozen PPO policy."""

    metadata = {"render_modes": []}

    def __init__(self, opponent_drone=None, degrade=False, sensor=None):
        self.observation_space = spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)  # az rate, el rate, fire>0
        self.opponent = opponent_drone
        self.world = World()
        self.sensor = sensor if sensor is not None else (Sensor() if degrade else None)
        self._freeze_drone = False  # test hook

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset(self.np_random)
        if self.sensor is not None:
            self.sensor.reset(self.world, self.np_random)
        return self.world.turret_obs(self.np_random, self.sensor), {}

    def _drone_accel(self):
        if self._freeze_drone:
            return np.zeros(3)
        if self.opponent is None:
            return self.world.scripted_accel()
        act, _ = self.opponent.predict(self.world.drone_obs(), deterministic=True)
        return np.clip(act, -1, 1) * DRONE_ACC

    def step(self, action):
        hit, reached, fired, nearmiss = self.world.step(action, self._drone_accel())
        reward = (-0.02 if fired else 0.0) + 0.3 * nearmiss
        ang = np.arccos(np.clip(np.dot(self.world.barrel(), _unit(self.world.drone)), -1, 1))
        reward -= 0.01 * ang  # aim-error shaping: learn tracking before lead
        terminated = False
        if hit:
            reward, terminated = reward + 10.0, True
        elif reached:
            reward, terminated = reward - 10.0, True
        trunc = self.world.t >= EP_LEN
        return self.world.turret_obs(self.np_random, self.sensor), reward, terminated, trunc, {"hit": hit}


class DroneEnv(gym.Env):
    """Agent = drone (attack the origin, dodge shots). Turret: frozen policy or passive (None)."""

    metadata = {"render_modes": []}

    def __init__(self, turret_model=None):
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)  # accel direction, scaled by DRONE_ACC
        self.turret_model = turret_model
        self.world = World()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset(self.np_random)
        return self.world.drone_obs(), {}

    def _turret_action(self):
        if self.turret_model is None:
            return np.zeros(3)  # passive turret (contract-test / null opponent)
        act, _ = self.turret_model.predict(self.world.turret_obs(self.np_random), deterministic=True)
        return act

    def step(self, action):
        drone_accel = np.clip(action, -1, 1) * DRONE_ACC
        hit, reached, _, _ = self.world.step(self._turret_action(), drone_accel)
        # mission: reach the asset alive
        reward = -0.01 - 0.02 * np.linalg.norm(self.world.drone) / 200.0
        terminated = False
        if hit:
            reward, terminated = reward - 10.0, True
        elif reached:
            reward, terminated = reward + 10.0, True
        trunc = self.world.t >= EP_LEN
        return self.world.drone_obs(), reward, terminated, trunc, {"reached": reached}


def hit_rate(turret_model, drone_model=None, episodes=100, degrade=False, sensor=None):
    """Turret interception rate against the given drone (scripted if None)."""
    env = TurretEnv(opponent_drone=drone_model, degrade=degrade, sensor=sensor)
    hits = 0
    for i in range(episodes):
        obs, _ = env.reset(seed=2000 + i)
        while True:
            act, _ = turret_model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                hits += info["hit"]
                break
    return hits / episodes


def test():
    from gymnasium.utils.env_checker import check_env
    check_env(TurretEnv())
    check_env(DroneEnv())  # null (passive) turret

    # geometry: frozen drone dead ahead, aim straight at it, fire -> must hit
    env = TurretEnv()
    env.reset(seed=0)
    env._freeze_drone = True
    env.world.drone = np.array([100.0, 0.0, 10.0])
    env.world.drone_vel = np.zeros(3)
    env.world.az, env.world.el = 0.0, np.arctan2(10.0, 100.0)
    assert any(env.step(np.array([0.0, 0.0, 1.0], np.float32))[4]["hit"] for _ in range(20)), \
        "point-blank ballistic shot missed a frozen drone"

    # both envs: random policy episodes must terminate
    for Env in (TurretEnv, DroneEnv):
        env = Env()
        for i in range(3):
            env.reset(seed=i)
            steps = 0
            while True:
                _, _, term, trunc, _ = env.step(env.action_space.sample())
                steps += 1
                if term or trunc:
                    break
            assert steps <= EP_LEN
    print("all checks passed")


def train(steps=300_000):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    model = PPO("MlpPolicy", make_vec_env(TurretEnv, n_envs=8), verbose=1)
    model.learn(total_timesteps=steps)
    model.save("turret_ppo")
    print(f"saved turret_ppo.zip | hit rate: {hit_rate(model):.2f}")


def evaluate():
    from stable_baselines3 import PPO
    print(f"hit rate: {hit_rate(PPO.load('turret_ppo')):.2f}")


def eval_perception(episodes=50):
    """End-to-end perception-in-the-loop: image -> CNN detector -> back-projected
    track -> the existing turret policy. The honest 'trained on images' number."""
    from stable_baselines3 import PPO
    turret = PPO.load("turret_ppo")
    truth = hit_rate(turret, episodes=episodes)
    percept = hit_rate(turret, episodes=episodes, sensor=DetectorSensor())
    print(f"perception-in-the-loop hit rate: {percept:.2f}  (vs {truth:.2f} ground-truth)")


def degrade_benchmark(steps=2_000_000):
    """Measure how the clean-trained turret holds up under realistic sensing
    (dropout/latency/range-noise), and whether naively retraining on that degraded
    sensor helps. Result (this setup): the clean policy degrades gracefully, and
    naive retraining is actually WORSE — dropout+latency make it a partially-observed
    problem a memoryless MLP can't crack without frame-history/RNN. The honest place
    to add realism is the sensor model, not the graphics."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    clean = PPO.load("turret_ppo")
    base = hit_rate(clean)                       # clean policy on clean sensing
    fragile = hit_rate(clean, degrade=True)      # clean policy on degraded sensing
    print(f"clean/clean {base:.2f} | clean/degraded {fragile:.2f}")

    model = PPO("MlpPolicy", make_vec_env(TurretEnv, n_envs=8,
                                          env_kwargs={"degrade": True}), verbose=1)
    model.learn(total_timesteps=steps)
    model.save("turret_degraded")
    robust = hit_rate(model, degrade=True)       # degraded-trained policy on degraded sensing
    print(f"degraded/degraded {robust:.2f}")

    import matplotlib.pyplot as plt
    labels = ["clean policy\nclean sensing", "clean policy\ndegraded sensing",
              "degraded-trained\ndegraded sensing"]
    vals, colors = [base, fragile, robust], ["tab:green", "tab:orange", "tab:red"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 5), ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("interception rate")
    ax.set_title("Realistic sensing: clean policy degrades gracefully;\nnaive retraining is worse (needs memory, not just noise)")
    fig.tight_layout()
    fig.savefig("degrade.png", dpi=120)
    print("saved degrade.png + turret_degraded.zip")


def selfplay(rounds=4, steps=500_000):
    """Alternate: train the stale side against the frozen fresh side, warm-started.
    Track the turret's hit rate each round -> arms-race curve."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    turret = PPO.load("turret_ppo")            # gen-0: already trained vs scripted
    drone = None
    hist = [("g0 turret\nvs scripted", hit_rate(turret))]
    print(f"g0: turret vs scripted -> {hist[0][1]:.2f}")

    for r in range(1, rounds + 1):
        if r % 2 == 1:  # drone's turn to adapt to the current turret
            venv = make_vec_env(DroneEnv, n_envs=8, env_kwargs={"turret_model": turret})
            if drone is None:
                drone = PPO("MlpPolicy", venv, verbose=0)
            else:
                drone.set_env(venv)
            drone.learn(total_timesteps=steps)
            label = f"g{r} drone\nadapts"
        else:           # turret's turn to adapt to the current drone
            turret.set_env(make_vec_env(TurretEnv, n_envs=8, env_kwargs={"opponent_drone": drone}))
            turret.learn(total_timesteps=steps)
            label = f"g{r} turret\nadapts"
        hr = hit_rate(turret, drone)
        hist.append((label, hr))
        print(f"round {r}: {label.replace(chr(10), ' ')} -> turret hit rate {hr:.2f}")

    turret.save("turret_selfplay")
    if drone is not None:
        drone.save("drone_selfplay")

    import matplotlib.pyplot as plt
    labels = [h[0] for h in hist]
    rates = [h[1] for h in hist]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(rates)), rates, "o-", lw=2)
    for i, (lab, rt) in enumerate(zip(labels, rates)):
        col = "tab:blue" if "turret" in lab else "tab:red"
        ax.scatter(i, rt, color=col, zorder=3, s=60)
        ax.annotate(f"{rt:.2f}", (i, rt), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("turret interception rate")
    ax.set_title("Self-play arms race: turret vs learning drone")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("selfplay.png", dpi=120)
    print("saved selfplay.png + turret_selfplay.zip / drone_selfplay.zip")


def _circle3d(c, n, r, k=14):
    """Ring of k points, radius r, centered at c in the plane with normal n."""
    n = _unit(n)
    u = _unit(np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0]))
    v = np.cross(n, u)
    th = np.linspace(0, 2 * np.pi, k)
    return c[:, None] + r * (np.outer(u, np.cos(th)) + np.outer(v, np.sin(th)))


def _rollout_hit(model, tries=25):
    """Run episodes until the turret scores an intercept, so the demo shows a kill.
    Records (drone, vel, barrel, shots[(pos,vel)]) per step. Falls back to last try."""
    env = TurretEnv()
    best = None
    for _ in range(tries):
        obs, _ = env.reset()
        frames = []
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            w = env.world
            frames.append((w.drone.copy(), w.drone_vel.copy(), w.barrel(),
                           [(p.copy(), vel.copy()) for p, vel, _ in w.shots]))
            if term or trunc:
                break
        best = frames
        if info["hit"]:
            return frames, True
    return best, False


def watch():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from stable_baselines3 import PPO

    frames, hit = _rollout_hit(PPO.load("turret_ppo"))
    print("HIT" if hit else "no intercept in sample; rendering last episode")
    n = len(frames)
    drones = np.array([f[0] for f in frames])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    SKY, GND = (0.53, 0.72, 0.90), (0.82, 0.80, 0.74)

    def draw(i):
        ax.clear()
        drone, vel, barrel, shots = frames[i]
        # sky + ground via pane colors and a ground grid
        for axis, col in ((ax.xaxis, SKY), (ax.yaxis, SKY), (ax.zaxis, GND)):
            axis.set_pane_color((*col, 1.0))
        gx = np.linspace(-160, 160, 9)
        for g in gx:
            ax.plot([g, g], [-160, 160], [0, 0], color=(0, 0, 0, 0.10), lw=0.6)
            ax.plot([-160, 160], [g, g], [0, 0], color=(0, 0, 0, 0.10), lw=0.6)

        # turret: base marker + barrel
        ax.scatter(0, 0, 0, c="k", marker="s", s=80)
        b = barrel * 18
        ax.plot([0, b[0]], [0, b[1]], [0, b[2]], "k-", lw=2)

        # drone trail
        tr = drones[max(0, i - 25):i + 1]
        if len(tr) > 1:
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=(0.8, 0.1, 0.1, 0.5), lw=1.5)

        # drone as a banking quad: 4 rotor disks on an X frame
        fwd = vel.copy(); fwd[2] = 0
        fwd = _unit(fwd) if np.linalg.norm(fwd) > 1e-6 else np.array([1.0, 0, 0])
        right = _unit(np.cross(fwd, [0, 0, 1.0]))
        accel = (vel - frames[i - 1][1]) if i > 0 else np.zeros(3)   # bank into the turn
        roll = float(np.clip(-0.04 * np.dot(accel, right), -0.6, 0.6))
        up = _unit(np.array([0, 0, 1.0]) * np.cos(roll) + right * np.sin(roll))
        right = _unit(np.cross(fwd, up))
        arm = 5.0
        for a, s in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            rc = drone + arm * (a * fwd + s * right)
            ax.plot([drone[0], rc[0]], [drone[1], rc[1]], [drone[2], rc[2]], "k-", lw=1)
            ring = _circle3d(rc, up, 2.2)
            ax.plot(ring[0], ring[1], ring[2], color="k", lw=1.2)
        ax.scatter(*drone, c="crimson", s=30)

        # projectile tracers
        for p, pv in shots:
            tail = p - _unit(pv) * 6
            ax.plot([tail[0], p[0]], [tail[1], p[1]], [tail[2], p[2]], color="orange", lw=2)

        # tracking camera: frame turret+drone, slow orbit
        c = drone / 2.0
        rad = max(60.0, np.linalg.norm(drone) * 0.6)
        ax.set(xlim=(c[0] - rad, c[0] + rad), ylim=(c[1] - rad, c[1] + rad), zlim=(0, max(60, drone[2] + 30)))
        ax.view_init(elev=22, azim=-60 + i * 0.35)
        ax.set_title(f"t = {i * DT:4.1f} s" + ("     ● INTERCEPT" if hit and i == n - 1 else ""))
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    seq = list(range(0, n, 2)) + [n - 1] * 6  # every other frame + linger on the kill
    imgs = []
    for i in seq:
        draw(i)
        fig.canvas.draw()
        rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        imgs.append(rgb.convert("P", palette=Image.ADAPTIVE, colors=96))
    imgs[0].save("episode.gif", save_all=True, append_images=imgs[1:],
                 duration=80, loop=0, optimize=True, disposal=2)
    print(f"saved episode.gif ({len(imgs)} frames)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == "train":
        train(arg or 300_000)
    elif cmd == "watch":
        watch()
    elif cmd == "eval":
        evaluate()
    elif cmd == "selfplay":
        selfplay(arg or 4)
    elif cmd == "degrade":
        degrade_benchmark(arg or 2_000_000)
    elif cmd == "perception":
        eval_perception(arg or 50)
    else:
        test()

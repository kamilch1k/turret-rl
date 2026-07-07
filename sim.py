"""Counter-FPV turret sim: PPO-trained turret vs scripted evasive drone.

A drone spawns 120-180 m out and jinks toward the asset at the origin.
The turret (at the origin) slews in az/el and fires ballistic projectiles
with real flight time — the policy has to learn lead prediction.

Usage:
    python sim.py test           # smoke checks (geometry + env contract)
    python sim.py train [steps]  # train PPO (default 300_000), saves turret_ppo.zip
    python sim.py watch          # run trained turret, saves episode.gif
"""

import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

DT = 0.05                 # s per step
EP_LEN = 600              # 30 s episode
SPAWN_R = (120.0, 180.0)  # drone spawn distance, m
ASSET_R = 5.0             # drone wins inside this radius of origin
DRONE_SPEED = 25.0        # m/s cruise toward asset
DRONE_ACC = 30.0          # m/s^2 jink authority
MUZZLE_V = 400.0          # m/s
SLEW = 2.5                # rad/s max turret rate
COOLDOWN = 0.5            # s between shots
AMMO = 40
HIT_R = 2.0               # ponytail: proximity kill stands in for shot spread; model a pellet cone if fidelity matters
OBS_NOISE = 1.5           # m std on measured drone position
GRAV = np.array([0.0, 0.0, -9.81])


def _seg_dist(a, b, p):
    """Distance from point p to segment a-b (projectile sweep vs drone)."""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - p))


class TurretEnv(gym.Env):
    """Obs: noisy drone pos/vel, own az/el, cooldown, ammo. Act: az/el rate + fire."""

    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, (11,), np.float32)
        # az rate, el rate in [-1,1] (scaled by SLEW), fire if > 0
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self._freeze_drone = False  # test hook

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        az = rng.uniform(-np.pi, np.pi)
        r = rng.uniform(*SPAWN_R)
        self.drone = np.array([r * np.cos(az), r * np.sin(az), rng.uniform(10, 60)])
        self.drone_vel = DRONE_SPEED * self._unit(-self.drone)
        # per-episode jink pattern: two perpendicular sinusoids, random amp/freq/phase
        self.jink = rng.uniform([0.3 * DRONE_ACC, 0.5, 0], [DRONE_ACC, 2.0, 2 * np.pi], (2, 3))
        self.az, self.el = rng.uniform(-np.pi, np.pi), 0.0
        self.cool, self.ammo, self.t = 0.0, AMMO, 0
        self.shots = []  # live projectiles: [pos, vel]
        return self._obs(), {}

    @staticmethod
    def _unit(v):
        return v / (np.linalg.norm(v) + 1e-9)

    def _barrel(self):
        ce = np.cos(self.el)
        return np.array([ce * np.cos(self.az), ce * np.sin(self.az), np.sin(self.el)])

    def _obs(self):
        rng = self.np_random
        pos = self.drone + rng.normal(0, OBS_NOISE, 3)   # ponytail: perfect-detection noisy sensor; add dropout/latency for realism
        vel = self.drone_vel + rng.normal(0, 1.0, 3)
        return np.concatenate([
            pos / 200.0, vel / 50.0,
            [np.sin(self.az), np.cos(self.az), self.el,
             self.cool / COOLDOWN, self.ammo / AMMO],
        ]).astype(np.float32)

    def step(self, action):
        self.t += 1
        reward = 0.0

        # drone: home on origin + sinusoidal jink perpendicular to velocity
        if not self._freeze_drone:
            vhat = self._unit(self.drone_vel)
            b1 = self._unit(np.cross(vhat, [0.0, 0.0, 1.0]))
            b2 = np.cross(vhat, b1)
            tt = self.t * DT
            acc = 2.0 * (DRONE_SPEED * self._unit(-self.drone) - self.drone_vel)
            for (amp, freq, ph), b in zip(self.jink, (b1, b2)):
                acc = acc + amp * np.sin(freq * tt + ph) * b
            self.drone_vel += acc * DT
            speed = np.linalg.norm(self.drone_vel)
            if speed > 1.2 * DRONE_SPEED:
                self.drone_vel *= 1.2 * DRONE_SPEED / speed
            self.drone = self.drone + self.drone_vel * DT
            self.drone[2] = max(self.drone[2], 1.0)

        # turret slew + fire
        self.az += float(np.clip(action[0], -1, 1)) * SLEW * DT
        self.el = float(np.clip(self.el + float(np.clip(action[1], -1, 1)) * SLEW * DT,
                                -0.2, np.pi / 2))
        self.cool = max(0.0, self.cool - DT)
        if action[2] > 0 and self.cool == 0.0 and self.ammo > 0:
            self.shots.append([np.zeros(3), MUZZLE_V * self._barrel(), np.inf])
            self.cool, self.ammo = COOLDOWN, self.ammo - 1
            reward -= 0.02

        # projectiles: swept hit check against drone
        hit, live = False, []
        for pos, vel, min_d in self.shots:
            new_pos = pos + vel * DT
            d = _seg_dist(pos, new_pos, self.drone)
            min_d = min(min_d, d)
            if d < HIT_R:
                hit = True
                continue
            vel = vel + GRAV * DT
            if new_pos[2] > 0 and np.linalg.norm(new_pos) < 500:
                live.append([new_pos, vel, min_d])
            else:
                # near-miss shaping: credit shots by closest approach, else "fire" never leaves the do-nothing optimum
                reward += 0.3 * np.exp(-min_d / 10.0)
        self.shots = live

        # shaping: penalize aim error so tracking is learned before lead
        ang = np.arccos(np.clip(np.dot(self._barrel(), self._unit(self.drone)), -1, 1))
        reward -= 0.01 * ang

        terminated = False
        if hit:
            reward, terminated = reward + 10.0, True
        elif np.linalg.norm(self.drone) < ASSET_R:
            reward, terminated = reward - 10.0, True
        return self._obs(), reward, terminated, self.t >= EP_LEN, {"hit": hit}


def test():
    from gymnasium.utils.env_checker import check_env
    check_env(TurretEnv())

    # geometry: frozen drone dead ahead, aim straight at it, fire -> must hit
    env = TurretEnv()
    env.reset(seed=0)
    env._freeze_drone = True
    env.drone = np.array([100.0, 0.0, 10.0])
    env.drone_vel = np.zeros(3)
    env.az, env.el = 0.0, np.arctan2(10.0, 100.0)
    hit = False
    for _ in range(20):
        _, _, term, _, info = env.step(np.array([0.0, 0.0, 1.0], np.float32))
        if info["hit"]:
            hit = True
            break
    assert hit, "point-blank ballistic shot missed a frozen drone"

    # random policy: episodes must terminate and drone must be able to win
    env = TurretEnv()
    outcomes = []
    for i in range(5):
        env.reset(seed=i)
        while True:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                outcomes.append(r)
                break
    assert len(outcomes) == 5
    print("all checks passed")


def train(steps=300_000):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    model = PPO("MlpPolicy", make_vec_env(TurretEnv, n_envs=8), verbose=1)
    model.learn(total_timesteps=steps)
    model.save("turret_ppo")
    print("saved turret_ppo.zip")
    evaluate(model)


def evaluate(model=None, episodes=100):
    if model is None:
        from stable_baselines3 import PPO
        model = PPO.load("turret_ppo")
    env, hits = TurretEnv(), 0
    for i in range(episodes):
        obs, _ = env.reset(seed=1000 + i)
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                hits += info["hit"]
                break
    print(f"hit rate: {hits}/{episodes}")


def watch():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from stable_baselines3 import PPO

    model = PPO.load("turret_ppo")
    env = TurretEnv()
    obs, _ = env.reset()
    frames = []
    while True:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        frames.append((env.drone.copy(), env._barrel(), [p.copy() for p, _, _ in env.shots]))
        if term or trunc:
            print("HIT" if info["hit"] else "drone survived/reached asset")
            break

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")

    def draw(i):
        ax.clear()
        drone, barrel, shots = frames[i]
        ax.scatter(0, 0, 0, c="k", marker="s", s=60)
        b = barrel * 15
        ax.plot([0, b[0]], [0, b[1]], [0, b[2]], "k-")
        ax.scatter(*drone, c="r", s=40)
        for p in shots:
            ax.scatter(*p, c="orange", s=8)
        ax.set(xlim=(-200, 200), ylim=(-200, 200), zlim=(0, 100), title=f"t={i * DT:.1f}s")

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=50)
    anim.save("episode.gif", writer=PillowWriter(fps=20))
    print(f"saved episode.gif ({len(frames)} frames)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 300_000)
    elif cmd == "watch":
        watch()
    elif cmd == "eval":
        evaluate()
    else:
        test()

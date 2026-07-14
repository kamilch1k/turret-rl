# turret-rl

A 3D counter-FPV simulation where both sides learn. A PPO **turret** learns lead
prediction against an evasive **drone**; the drone learns to evade back via
self-play; and a separate CNN handles the "find the drone in a noisy image"
problem. Fire control and perception are modeled as the *separate stages* they
are in a real counter-UAS system — not fused into one magic pixel-to-action box.

![demo](demo.gif)

*PPO turret (black, at origin) intercepting an evasive quad. Red = drone trail, orange = projectile tracers. Camera tracks and zooms toward the kill.*

## Results

**Turret vs scripted drone — 82/100 interceptions** (2M steps, ~25 min on one laptop).
It learns to lead a fast, jinking target under noisy sensing, given projectile
flight time, limited slew rate, cooldown, and finite ammo.

**Self-play arms race.** Make the drone its own PPO agent (`DroneEnv`) and
alternate training each side against the frozen other, warm-started from the 82%
turret. The turret's interception rate sawtooths — the drone learns to evade and
collapses it to ~0.02, the turret re-adapts to ~0.75, repeat:

![arms race](selfplay.png)

It *oscillates* rather than settling: with deterministic best-response, whoever
adapts last dominates. That's a real property of naive self-play — real systems
damp it with population/league play (a documented next step, not a bug).

**Sensing realism beats graphics — with a twist.** A policy is only as good as
the track it's fed. Under a realistic sensor model — ~300 ms pipeline latency,
range-dependent dropout, range-scaled noise — the clean-trained turret degrades
*gracefully*: **0.81 → 0.69**. Counterintuitively, a turret trained *directly on*
that degraded sensor did **worse (0.38)**: heavy dropout + latency turn it into a
partially-observed problem a memoryless MLP can't crack, so 2M steps of noisier
signal produced a weaker policy. Robustness here came from clean training
generalizing — *not* from "training on noise." Getting a win from degraded-sensor
training would need memory (frame-stacking / RNN) or a clean→degraded curriculum.
That's a real experimental result, reported as it came out:

![sensing](degrade.png)

**Perception breaks with range.** The turret consumes a *track*, not pixels — the
"find the drone" problem is a separate detector. A small CNN on synthetic camera
frames holds ~90% detection in a 35–65 m sweet spot and collapses past ~70 m as
the drone shrinks toward a single low-contrast pixel and sinks into sensor noise:

![detection](detection.png)

## How the stages fit

```
camera frame  ->  CNN detector  ->  track (pos, vel)  ->  PPO fire control  ->  shot
                  (detect.py)       degraded sensor        (sim.py)
```

This decomposition is why an abstract sim is legitimate: the controller never
sees pixels even in a real deployment, so prettier graphics would not change the
policy — only a more realistic *observation model* would. Realism lives in the
sensor, not the render.

## Run

```
pip install gymnasium stable-baselines3 matplotlib torch
python sim.py test               # physics + both env contracts
python sim.py train [steps]      # turret vs scripted drone -> turret_ppo.zip
python sim.py eval               # interception rate over 100 episodes
python sim.py watch              # render an episode -> episode.gif
python sim.py selfplay [rounds]  # arms race -> selfplay.png
python sim.py degrade [steps]    # perfect-vs-degraded-sensing benchmark -> degrade.png
python detect.py                 # perception stage -> detection.png
```

## Scope & non-goals

Deliberate boundaries, stated up front:

- **The controller is state-based, not pixel-trained.** End-to-end pixel RL for
  fire control won't converge on commodity hardware (rendering in the training
  loop drops throughput ~100×) and isn't how real systems work — the controller
  consumes tracks. Images live in the detector stage. That's the correct
  architecture, not a shortcut.
- **No Unreal/AirSim photorealism.** It wouldn't change the trained policy; the
  renderer is a stylized demo on purpose.
- **Physics is point-mass**, not rotor-level 6-DOF — it captures the target
  motion envelope the turret cares about, not blade aerodynamics.
- **A fuller perception pipeline would** train the detector on real anti-UAV
  imagery (Anti-UAV, Drone-vs-Bird) instead of synthetic frames, add clutter and
  false-positive metrics, and fuse detector output into the live control loop.

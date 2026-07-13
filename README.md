# turret-rl

3D counter-FPV turret simulation. A PPO-trained turret (azimuth/elevation slew +
ballistic projectiles with real flight time) learns **lead prediction** against a
scripted evasive drone under noisy sensing.

![demo](demo.gif)

*Red = evasive drone, black = turret, orange = projectiles. Origin is the defended asset.*

**Result:** 82/100 interceptions over 100 held-out episodes after 2M training steps
(~25 min on one laptop GPU).

```
pip install gymnasium stable-baselines3 matplotlib
python sim.py test           # smoke checks
python sim.py train [steps]  # train PPO, saves turret_ppo.zip + prints hit rate
python sim.py eval           # hit rate over 100 episodes
python sim.py watch          # render an episode to episode.gif
```

## What it models

Fire control, not perception. The policy consumes a *track* (drone position/velocity
plus Gaussian sensor noise), not pixels — the same interface a real detector/tracker
would feed it. The hard part it learns is leading a fast, jinking target given
projectile flight time, limited slew rate, cooldown, and finite ammo.

Next: a degraded-sensor observation model (detection dropout, latency,
range-dependent noise) with a perfect-vs-degraded benchmark; then an RL drone
trained via self-play.

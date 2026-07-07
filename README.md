# turret-rl

3D counter-FPV turret simulation. A PPO-trained turret (azimuth/elevation slew +
ballistic projectiles with real flight time) learns lead prediction against a
scripted evasive drone under noisy sensing.

```
pip install gymnasium stable-baselines3 matplotlib
python sim.py test           # smoke checks
python sim.py train [steps]  # train PPO, saves turret_ppo.zip + prints hit rate
python sim.py eval           # hit rate over 100 episodes
python sim.py watch          # render an episode to episode.gif
```

Status: WIP — env + training pipeline verified end-to-end; tuning toward a
useful hit rate. Next: degraded-sensor observation model (dropout, latency,
range-dependent noise) and perfect-vs-degraded benchmark; then RL drone via
self-play.

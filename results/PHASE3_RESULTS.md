# Phase 3 Results — Connectome-Constrained vs. Baseline Network

## Hypothesis

Does a PyTorch network whose connectivity is masked to match a real fly
optomotor circuit (T4/T5 → HS/VS → descending neurons, extracted from
FlyWire in Phase 1) learn a simple 2D pig-dodge steering task faster or
better than a same-sized, same-edge-count, randomly-rewired network?

## Method

- **Task:** `PigDodgeEnv` (Gymnasium) — a pig sprite moves left/right/stays
  to dodge a falling stimulus. +0.05/step survival, +1.0 per successful
  dodge, -10.0 on collision (episode ends).
- **Networks:** both variants share identical neuron counts per layer
  (12,245 input / 38 processing / 93 output) and identical edge counts per
  layer (6,836 / 12,135 / 359). The *connectome* network's edges are the
  real synapses extracted from FlyWire; the *baseline* network's edges are
  randomly rewired subject to the same counts. Both networks are otherwise
  identical (same sensor/motor read-in/read-out layers, same activation,
  same sparse-matmul implementation).
- **Training:** PPO (Stable-Baselines3), 150,000 timesteps per run,
  3 seeds (0, 1, 2) per variant — 6 runs total.
- **Metric:** final episode reward, computed as the mean of the last 20
  completed episodes in each run.

## Results

| Variant    | Seed 0 | Seed 1 | Seed 2 | Mean  | Std  |
|------------|--------|--------|--------|-------|------|
| Connectome | 22.21  | 26.42  | 26.63  | 25.09 | 2.49 |
| Baseline   | 20.52  | 17.66  | 30.73  | 22.97 | 6.87 |

Full per-run logs: `data/processed/logs/{connectome,baseline}_seed{0,1,2}/monitor.monitor.csv`
Raw numbers: `data/processed/results_summary.csv`
Reward curves (mean ± std band across seeds): `data/processed/reward_comparison.png`
Trained-pig behavior: `data/processed/pig_connectome_trained.gif`, `data/processed/pig_baseline_trained.gif`
Untrained baseline for comparison: `data/processed/pig_random.gif`

## Honest interpretation

The connectome-constrained network finished with a **higher mean** final
reward (25.09 vs. 22.97) and **notably lower variance** across seeds (2.49
vs. 6.87) — the baseline's seed 1 badly underperformed while its seed 2
badly overperformed, whereas all three connectome seeds landed in a tight
band. That consistency is the more interesting signal here than the mean
gap itself.

That said, this is a modest result that should not be oversold:

- The mean reward curves (`reward_comparison.png`) **overlap heavily**
  for most of training — the two variants are frequently
  indistinguishable within a standard-deviation band until late training.
- **3 seeds per variant is limited statistical power.** RL is noisy by
  nature; a difference of this size with this few seeds is suggestive,
  not conclusive.
- The task (dodging a 1D falling stimulus) is a deliberately simple proxy
  for the kind of motion-detection behavior the real T4/T5→HS/VS→DN
  circuit evolved for — it is not a claim that this replicates real fly
  behavior, only that the circuit's *topology* transfers to an analogous
  steering problem.

**Conclusion:** the real fly circuit topology is not obviously worse than
a random network of the same size at this task, and shows more consistent
learning across seeds. That is a legitimate, modest, postable finding —
consistent with the plan's own honesty note that either outcome (win or
lose) is worth reporting as long as it's reported accurately. It does not
establish that the connectome topology is a generically better
architecture; more seeds, task variants, or ablations would be needed to
make a stronger claim.

## What would strengthen this result

- More seeds (5–10 per variant) to get a real confidence interval on the
  mean-reward gap.
- A second task variant (e.g. 2D stimulus motion, faster stimuli) to test
  whether the topology's advantage (if any) is task-specific.
- Sample-efficiency comparison (steps to reach a reward threshold), not
  just final reward.

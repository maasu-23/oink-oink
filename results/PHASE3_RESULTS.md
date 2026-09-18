# Phase 3 Results — Connectome-Constrained vs. Baseline Network

## Hypothesis

Does a PyTorch network whose connectivity is masked to match a real fly
optomotor circuit (T4/T5 → HS/VS → descending neurons, extracted from
FlyWire in Phase 1) learn a simple 2D pig-dodge steering task faster or
better than a same-sized, same-edge-count, randomly-rewired network?

## Method

- **Task:** `PigDodgeEnv` (Gymnasium) — a pig moves left/right/stays to
  dodge a ball thrown at it from above. +0.05/step survival, +1.0 per
  successful dodge, −10.0 on collision (episode ends). Episodes cap at
  500 steps, so ~31–32 is the practical maximum reward.
- **Networks:** both variants share identical neuron counts per layer
  (12,245 input / 38 processing / 93 output) and identical edge counts per
  layer (6,836 / 12,135 / 359). The *connectome* network's edges are the
  real synapses extracted from FlyWire; the *baseline* network's edges are
  randomly rewired subject to the same counts. Both networks are otherwise
  identical (same sensor/motor read-in/read-out layers, same activation,
  same sparse-matmul implementation).
- **Training:** PPO (Stable-Baselines3), 150,000 timesteps per run,
  3 seeds (0, 1, 2) per variant — 6 runs total.
- **Metrics:** final episode reward (mean of the last 20 episodes),
  sample efficiency (timesteps until the 20-episode rolling mean first
  reaches 25), and reliability (how many seeds learned at all).

## Results (v2 — ball thrown at the pig)

| Variant    | Seed 0 | Seed 1 | Seed 2 | Mean  | Std   |
|------------|--------|--------|--------|-------|-------|
| Connectome | 31.74  | 31.60  | 31.55  | 31.63 | 0.10  |
| Baseline   | 29.28  | 2.65   | 30.16  | 20.69 | 15.64 |

Sample efficiency — timesteps until the rolling-mean reward first reached 25:

| Variant    | Seed 0 | Seed 1 | Seed 2 |
|------------|--------|--------|--------|
| Connectome | 23,888 | 27,435 | 22,435 |
| Baseline   | 39,154 | never  | 42,775 |

Share of the final 50 episodes in which the pig survived the full 500
steps: connectome 0.96 / 0.94 / 0.98; baseline 0.94 / 0.00 / 0.90.

Files:
- Reward curves (mean ± std across seeds): `data/processed/reward_comparison.png`
- Raw numbers: `data/processed/results_summary.csv`
- Per-run logs: `data/processed/logs/{connectome,baseline}_seed{0,1,2}/monitor.monitor.csv`
- Trained-pig behavior: `data/processed/pig_connectome_trained.gif`, `data/processed/pig_baseline_trained.gif`
- Untrained pig for comparison: `data/processed/pig_random.gif`

## Honest interpretation

Three findings, in decreasing order of confidence:

1. **The connectome network learned faster.** All three connectome seeds
   crossed a rolling reward of 25 by ~22–27k timesteps; the two baseline
   seeds that learned at all needed ~39–43k — roughly 1.7× as many
   samples. This is the clearest signal in the experiment and is visible
   directly in the reward curves.
2. **The connectome network learned more reliably.** 3/3 connectome seeds
   converged, with final rewards within 0.2 of each other. 2/3 baseline
   seeds converged; baseline seed 1 never escaped the "get hit quickly"
   regime in 150k steps. The baseline's high std (15.6) is almost entirely
   that one failed seed.
3. **Final performance among runs that converged is similar.** The two
   successful baseline seeds ended at 29.3 and 30.2 vs. the connectome's
   ~31.6 — a real but small gap, near the task's ceiling. Both successful
   policies converged to essentially the same rule (always move away from
   the ball, never stand still) and agree on ~91% of random inputs.

Caveats that still apply:

- **3 seeds per variant is limited statistical power.** "2/3 vs 3/3 seeds
  converged" could be luck; 10+ seeds would be needed to make the
  reliability claim firmly.
- The task is a deliberately simple proxy for the motion-detection
  behavior the real T4/T5 → HS/VS → DN circuit evolved for. The result
  says the circuit's *topology* is a useful prior for an analogous 1-D
  steering problem — not that it replicates fly behavior.
- The connectome network also inherits the real synapse *counts* as its
  initial weights (rescaled), whereas the baseline starts from random
  weights on random edges. The experiment therefore tests
  "real topology + real weights" vs. "random topology + random weights";
  it does not separate the contribution of topology from that of
  initialization. An ablation (real topology, random weights) would be
  the natural next step.

**Conclusion:** on this task, the real fly circuit topology learned roughly
1.7× faster than a size-matched random network and converged on every
seed where the baseline failed on one. It is a modest, suggestive result
rather than a definitive one, and it should be reported that way.

## What changed from v1 (and why)

The first version of this experiment (`data/processed/v1_uniform_spawn/`)
spawned the ball at a *uniformly random* x-position. Watching the trained
pigs revealed that every policy — connectome and baseline alike — had
learned to walk to a corner and stand still: with a 40px hit-zone on a
400px screen, a stationary pig already dodged ~90% of balls by luck, so
PPO never had a reason to learn real dodging. The v1 numbers
(connectome 25.09 ± 2.49 vs baseline 22.97 ± 6.87) were real, but they
measured "who camps more efficiently," not "who dodges better."

The fix (`oink/env.py::_spawn_stimulus`) throws the ball *at* the pig: it
spawns near the pig's current x and is aimed at that position with some
noise. Standing still now gets hit within ~75–135 steps, and even a naive
"step away from the ball" heuristic survives 4× longer, so dodging is the
only way to score. Everything else — networks, PPO hyperparameters,
seeds, timestep budget — is unchanged, so the v1 and v2 runs are directly
comparable. The v1 artifacts are kept for transparency.

## What would strengthen this result

- 10+ seeds per variant, to put a confidence interval on "how often does
  the baseline fail to learn."
- The topology-vs-initialization ablation described above.
- A harder task variant (faster balls, two balls at once, 2-D motion) to
  see whether the sample-efficiency gap widens or closes.

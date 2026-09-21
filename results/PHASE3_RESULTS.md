# Phase 3 Results — Connectome-Constrained vs. Baseline Network

## Hypothesis

Does a PyTorch network whose connectivity is masked to match a real fly
optomotor circuit (T4/T5 → HS/VS → descending neurons, extracted from
FlyWire in Phase 1) learn a simple 2D pig-dodge steering task faster or
better than a same-sized, same-edge-count, randomly-rewired network?

## Method

- **Task:** `PigDodgeEnv` (Gymnasium) — a pig moves left/right/stays to
  dodge a pointed dripstone falling at it from above (drawn as a red ball
  in v1; the sprites are cosmetic and do not affect training). +0.05/step survival, +1.0 per
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

## Results (v2 — stimulus thrown at the pig)

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

## Results (v3 — SageMaker replication + topology-vs-weights ablation)

The v2 comparison was re-run as 9 SageMaker training jobs (`ml.m5.xlarge`,
ap-south-1, ~2.5 h each; launched with `scripts/sagemaker_train.py`), using
the same seeds, hyperparameters and timestep budget, plus a third variant:

- **connectome_randinit** — the real wiring (same 6,836 / 12,135 / 359
  edges) but *random* initial weights on those edges, instead of the
  rescaled synapse counts. This separates "topology" from
  "topology + initialization".

| Variant (SageMaker)  | Seed 0 | Seed 1 | Seed 2 | Mean  | Std   | Reached 25 |
|----------------------|--------|--------|--------|-------|-------|------------|
| Connectome           | 26.30  | 27.34  | 21.56  | 25.07 | 3.08  | 3/3 (67k / 16k / 33k) |
| Connectome, rand init| −2.33  | 28.18  | −2.30  | 7.85  | 17.61 | 1/3 (— / 49k / —) |
| Baseline             | 31.79  | −2.06  | 29.00  | 19.58 | 18.79 | 2/3 (37k / — / 66k) |

Files: `data/processed/sagemaker/{reward_comparison.png,results_summary.csv,logs/,models/}`.

**The same seeds did not reproduce the laptop trajectories.** Connectome
seed 0 crossed 25 at 24k steps on the laptop and at 67k on SageMaker; seed 1
at 27k vs 16k. Different CPUs give different floating-point rounding in the
sparse matmuls, and PPO amplifies tiny differences into different training
histories. This is expected for RL and is exactly why the laptop numbers
(a tight 31.6 ± 0.1) should not have been read as precise. Treat the two
runs as independent replicates — 6 seeds per variant in total.

### Pooled (laptop v2 + SageMaker v3, 6 seeds per variant)

| Variant              | n | Final reward   | Seeds that learned | Steps to reach 25 (median of learners) |
|----------------------|---|----------------|--------------------|----------------------------------------|
| Connectome           | 6 | 28.4 ± 4.0     | 6/6                | ~26k |
| Baseline             | 6 | 20.1 ± 15.4    | 4/6                | ~41k |
| Connectome, rand init| 3 | 7.9 ± 17.6     | 1/3                | 49k (one seed) |

## Honest interpretation

Three findings, in decreasing order of confidence:

1. **The real circuit (wiring + synapse strengths) learns more reliably.**
   6/6 connectome seeds learned the task across two independent runs on
   different hardware; 4/6 baseline seeds did. The baseline's failures are
   total (the pig never escapes the "get hit quickly" regime), which is why
   its std is so large.
2. **It learns faster, but by less than v2 alone suggested.** Median time to
   a rolling reward of 25 is ~26k steps vs ~41k for the baseline — about
   1.5× (v2 alone said 1.7×). The spread is wide (16k–67k for the
   connectome), so this is a trend, not a tight estimate.
3. **The wiring alone is not the source of the advantage.** With random
   initial weights on the real edges, only 1/3 seeds learned — *worse* than
   the randomly wired baseline (2/3). The real synapse counts, used as
   initial weights, are doing most of the work: they hand PPO a network
   that already roughly maps "stimulus moving left" to "turn right." This
   is arguably the more interesting result: the connectome's value here
   comes from its *strengths*, not just its *shape*. Three seeds, so it
   too is suggestive rather than settled.

Among runs that converged, final performance is similar and near the
task's ceiling (~31–32); converged policies of every variant use the same
rule (always step away from the falling stimulus).

Caveats that still apply:

- **6 seeds (3 for the ablation) is still limited statistical power.**
  "4/6 vs 6/6" could be luck; 10+ seeds per variant would be needed to make
  the reliability claim firmly.
- The task is a deliberately simple proxy for the motion-detection
  behavior the real T4/T5 → HS/VS → DN circuit evolved for. The result
  says the circuit is a useful *prior* for an analogous 1-D steering
  problem — not that it replicates fly behavior.
- Some SageMaker connectome seeds drifted downward late in training
  (seed 2 ended at 21.6 after peaking above 30). PPO with these
  hyperparameters is not perfectly stable at 150k steps; the "final = mean
  of last 20 episodes" metric is sensitive to that.

**Conclusion:** on this task, the real fly circuit — wiring *and* synapse
strengths — learned on every seed and roughly 1.5× faster than a
size-matched random network, while the wiring with random strengths did
not help at all. It is a modest, suggestive result rather than a
definitive one, and it should be reported that way.

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

- 10+ seeds per variant (cheap now: `scripts/sagemaker_train.py launch
  --seed 0 1 2 ... 9` runs them in parallel), to put a confidence interval
  on "how often does the baseline fail to learn."
- The mirror ablation: *random* wiring with the real synapse-count
  *distribution* as initial weights, to check whether it is the weight
  magnitudes or their placement on specific edges that matters.
- A harder task variant (faster stimuli, two at once, 2-D motion) to
  see whether the sample-efficiency gap widens or closes.

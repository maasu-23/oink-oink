# PigBrain — Fly-Connectome-Constrained RL Agent

## Concept

A reinforcement learning agent whose neural network topology is built from a real, published circuit in the female fruit fly connectome (FlyWire), trained via RL on a toy task, visualized as a pig sprite. A Bedrock agent supervises training and can answer questions about the underlying biology.

**Hypothesis:** Does a network shaped like a real fly visual/optomotor circuit learn a simple steering task faster or better than a same-sized generic (baseline) network?

**Honesty note for writeups:** The RL agent is the part that is genuinely self-learning (improves through trial and error). The Bedrock layer is an automated training *supervisor* — it makes decisions about the training process, but does not itself learn over time unless explicitly built to. Keep this distinction clear in the README and any posts.

---

## Phase 0 — Lock the spec

- **Circuit:** visual motion / optomotor circuit (input visual neurons → processing → motor output neurons).
- **Task/environment:** 2D Pygame/Gymnasium environment. A pig sprite (Minecraft-style or simple pixel art) steers toward or away from a moving stimulus (e.g. dodge a thrown object, or chase a moving food item), controlled by the network's output.
- **Deliverable:** one-paragraph spec (circuit + task + hypothesis) and a rough sketch/mockup of the pig environment.

## Phase 1 — Extract the circuit topology (~1 week)

- Pull neuron IDs and synapse table for the visual/optomotor region from FlyWire/Codex.
- Build an adjacency matrix: rows/columns = neurons, values = synapse weight (count) and sign (excitatory/inhibitory, from neurotransmitter type where available).
- Store raw extract + processed adjacency matrix in S3 (e.g. `.npz` or Parquet).
- Sanity-check: input neuron count roughly matches visual neurons, output count matches motor neurons. If not, revisit the circuit choice before moving on.
- **Deliverable:** extraction script/notebook + a subgraph visualization image.

## Phase 2 — Build the pig environment + networks (~1.5 weeks)

- Build the Pygame/Gymnasium environment: pig sprite, moving stimulus, reward signal (+reward for correct dodge/chase, -reward for collision or miss).
- Build the connectome-constrained network: custom PyTorch module, weights masked to only exist where real synapses exist (from the Phase 1 adjacency matrix).
- Build a baseline network: standard/randomly-sparse network with the **same neuron count and same connection count** as the real circuit — this is the fair control group.
- Wire both into a PPO (or similar) training loop on SageMaker, driving the pig's actions in the environment.
- **Deliverable:** GIF of the untrained pig behaving randomly.

## Phase 3 — Run the experiment (~3-5 days)

- Train both networks (connectome-constrained vs. baseline), multiple seeds each — RL is noisy, so single runs prove nothing.
- Compare: final reward, sample efficiency, training stability.
- Capture GIFs of pig behavior at early and converged checkpoints for both networks.
- Be prepared for either result — if the real topology underperforms, that's still a legitimate, postable finding (e.g. "structure evolved for a different ecological task").
- **Deliverable:** results writeup with plots + trained-pig GIF.

## Phase 4 — Bedrock supervisor layer (~1 week)

- Bedrock agent with tool access to:
  - live training metrics
  - the connectome graph data
  - ability to trigger a retrain with adjusted hyperparameters
- Job: monitor runs, decide when to stop/retrain, and answer questions like "why does this circuit struggle with fast stimuli?" by referencing the actual synapse data.
- Optional: have it narrate the pig's progress in its explanations for a more engaging demo — cosmetic only, no impact on the actual ML.
- Described precisely as an **automated training supervisor** in all writeups, not as something that learns itself.
- **Deliverable:** short demo video of the agent narrating a training run over pig footage.

## Phase 5 — Package for LinkedIn / Builder Center (~2-3 days)

- Architecture diagram: S3 → extraction script → SageMaker (training) → Bedrock (supervisor) → pig demo/dashboard.
- Core visuals: subgraph image, untrained-vs-trained pig GIF, one reward-curve chart.
- README: hypothesis, method, and honest result stated up front.
- Repo/post name: **PigBrain** (or similar).

---

## AWS Architecture Summary

| Component | Service |
|---|---|
| Raw + processed connectome data | S3 |
| Circuit topology extraction | Script/notebook (optionally Glue/PySpark if scaling up) |
| RL training (pig environment) | SageMaker |
| Training supervisor / Q&A agent | Bedrock |
| Orchestration / demo interface | Lambda + API Gateway |
| Metrics/results dashboard | QuickSight or simple custom dashboard |

## Timeline

~4-5 weeks part-time, with postable milestones after:
- **Phase 1** — subgraph visualization
- **Phase 3** — trained pig GIF + results comparison
- **Phase 5** — full package (LinkedIn + AWS Builder Center)

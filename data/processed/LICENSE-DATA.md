# Data licence and attribution

Everything in this directory is derived from the **FlyWire** whole-brain
connectome of an adult female *Drosophila melanogaster* (FAFB, materialization
v783), accessed through FlyWire Codex. FlyWire's public release data is made
available under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial
4.0), and so are these derived files:

- `circuit_nodes.csv`, `circuit_adjacency.npz`, `circuit_subgraph.png`,
  `circuit_3d.gif` — the extracted T4/T5 → HS/VS → DN circuit, its signed
  synapse-count adjacency matrix, and renderings of the real neuron skeletons.
- `models/`, `sagemaker/models/` — trained policies whose connectome-variant
  weights are initialised from the real synapse counts.
- `pigbrain_dopamine.mp4` — renders FlyWire neuron skeletons (including PAM
  dopaminergic neurons) coloured by network activity.
- Training logs, plots and CSVs describing the above.

Non-commercial use only; attribute FlyWire. The code in this repository is
separately licensed under MIT (see the top-level `LICENSE`).

## Please cite

- Dorkenwald, S. *et al.* Neuronal wiring diagram of an adult brain.
  *Nature* 634, 124–138 (2024). https://doi.org/10.1038/s41586-024-07558-y
- Schlegel, P. *et al.* Whole-brain annotation and multi-connectome cell
  typing of *Drosophila*. *Nature* 634, 139–152 (2024).
  https://doi.org/10.1038/s41586-024-07686-5
- Matsliah, A. *et al.* Neuronal parts list and wiring diagram for a visual
  system. *Nature* 634, 166–180 (2024).
  https://doi.org/10.1038/s41586-024-07981-1 (source of the T4/T5 and HS/VS
  cell-type annotations used here)
- Zheng, Z. *et al.* A complete electron microscopy volume of the brain of
  adult *Drosophila melanogaster*. *Cell* 174, 730–743 (2018).
  https://doi.org/10.1016/j.cell.2018.06.019 (the FAFB EM volume)

FlyWire: https://flywire.ai · Codex: https://codex.flywire.ai

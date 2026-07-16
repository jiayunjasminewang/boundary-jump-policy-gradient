# Boundary-Jump Policy Gradient for Multiclass Queueing Networks

This repository contains the research code for studying boundary-jump
corrections in actor--critic policy gradients for reflected Brownian motion
and multiclass, multi-pool queueing systems. The experiments compare otherwise
identical training runs with and without the jump correction across 2-, 6-,
30-, and 100-dimensional systems.

The repository accompanies a working paper by the author. A citation entry will be added when a public preprint is available.

## Main contributions

- A boundary-jump correction for the reflected adjoint process.
- Joint learning of routing matrices and diffusion-scale intensity controls.
- Heterogeneous multiclass queueing configurations from 2 to 100 dimensions.
- Common-random-number Monte Carlo evaluation of learned policies.

## Repository structure

```text
.
├── main.py                 # Training entry point
├── solver.py               # Actor--critic models and training procedure
├── adjoint.py              # Reflected adjoint and boundary-jump correction
├── equation.py             # Queueing/RBM dynamics and costs
├── utils.py                # Numerical utilities
├── configs/                # Jump and no-jump experiment configurations
├── eval_compare_2d.py        # 2D Monte Carlo comparisons
├── eval_compare_6d.py        # 6D Monte Carlo comparisons
├── eval_compare_30d.py       # 30D Monte Carlo comparisons
└── eval_compare_100d.py      # Batched 100D Monte Carlo comparisons
```

Generated samples, checkpoints, and model weights are not included in the
repository. By default, training writes generated data and learned weights to
local output paths used by the solver.

## Installation

Python 3.10--3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Training

Each experiment has a jump configuration and a corresponding `_nojump`
configuration. The `train_config.use_jump` field is the switch that enables or
disables the boundary-jump correction.

For example, train the 2D symmetric system with the correction using:

```bash
python main.py --config_path=configs/queue_2d_symmetric.json
```

Train its no-jump counterpart using:

```bash
python main.py --config_path=configs/queue_2d_symmetric_nojump.json
```

The same pattern applies to the remaining configurations:


| Dimension | Regimes                             |
| --------- | ----------------------------------- |
| 2D        | `symmetric`, `graded`, `asymmetric` |
| 6D        | `symmetric`, `graded`, `asymmetric` |
| 30D       | `ratio100`, `ratio105`, `ratio200`  |
| 100D      | `ratio100`, `ratio105`, `ratio200`  |


Set `--dump=False` to load previously generated training samples rather than
generating and saving a new sample set:

```bash
python main.py \
  --config_path=configs/queue_30d_ratio105.json \
  --dump=False
```

Random seeds are read from the configuration files and applied to Python,
NumPy, and TensorFlow.

## Evaluation

The evaluation scripts are included. They perform:

1. controlled $2\times2$ Monte Carlo cross-evaluation using common random
  numbers;
2. the deployment comparison given by the diagonal of that table; and
3. a critic-free comparison under $\hat u\equiv0$.

The controlled simulation includes a terminal-value bootstrap at the end of
the Monte Carlo horizon. The critic-free zero-control comparison does not use
a value network and is therefore reported as a finite-horizon routing
diagnostic.

Before running an evaluation script, set its `CONFIG`, `STEM_JUMP`, and
`STEM_NOJUMP` variables to the configuration and checkpoints produced by the
corresponding training runs. For example:

```bash
python eval_compare_30d.py
```

The 2D, 6D, and 30D evaluations use 40,000 Monte Carlo trajectories per policy. Because of the substantially higher computational cost, the 100D evaluation uses 8,000 trajectories per policy. All policy comparisons within the same configuration use common random numbers.

The 100D implementation processes trajectories in batches to limit memory
usage; its batch size can be changed with:

```bash
EVAL100_BATCH_SIZE=128 python eval_compare_100d.py
```

For high-dimensional experiments, the three runs correspond to:

- `ratio100`: $r=1.00$ (flat);
- `ratio105`: $r=1.05$ (near-critical);
- `ratio200`: $r=2.00$ (wide).

## Reproducibility notes

- Jump and no-jump runs should use the matching pair of configuration files.
- Evaluation pairs must use checkpoints from the corresponding regime.
- Common random numbers are used within each Monte Carlo comparison.
- The 100D evaluator is batched but resets the same random seed for each policy
so that paired comparisons use the same noise sequence.
- Exact floating-point results may vary slightly across TensorFlow versions,
hardware, and oneDNN/CUDA execution paths.

## Code provenance and acknowledgments

This repository is derived from
[RBMSolver](https://github.com/nian-si/RBMSolver), permission was granted by the author to release this modified implementation under a license chosen by the present author.

RBMSolver was itself adapted from  
[DeepPDE_ActorCritic](https://github.com/MoZhou1995/DeepPDE_ActorCritic),  
which is distributed under the MIT License.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
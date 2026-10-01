# Deep RL Inventory Restocking

This repository evaluates a NumPy DQN inventory policy against Fixed Reorder, EOQ, and tuned `(s,S)` baselines under Poisson demand, lead times, capacity limits, and explicit economic costs. The environment follows the Gymnasium API shape but does not depend on Gymnasium or subclass `gym.Env`.

## Setup

Use Python 3.11.9, create a virtual environment, and install the pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce

Run the full pipeline with five seeds, 500 training episodes, and 200 paired evaluation episodes:

```bash
python reproduce.py
```

For a smoke run:

```bash
python reproduce.py --quick
```

The run preserves experiment subdirectories, refreshes only the main-run files, and tees console output to `results/run_log.txt`. The CPU name is recorded in `results/env_info.json`. Final DQN hyperparameters are frozen in `config.py`; test seeds begin at 1,000,000 and tuning seeds use a separate validation range.

## Experiment Commands

```bash
python experiments.py --ablation
python experiments.py --sensitivity
python experiments.py --nonstationary
```

Non-stationary evaluations include both a stationary-tuned zero-shot `(s,S)` comparator and a policy retuned on the target demand. Trend and regime-shift specialists include the demand-average state feature. Cost scenarios keep the selling price at ₹400 while scaling unit cost, holding cost, and stockout cost.

## Runtime And Outputs

Runtime depends on CPU and episode counts; `--quick` is the inexpensive validation path. The main run writes:

- `evaluation_results.csv`, including `discarded_stock` per episode
- `tuned_ss.json`, `summary_statistics.csv`, `per_seed_profit.csv`, and `hypothesis_tests.json`
- `fig1` through `fig6` as PDF and PNG files
- `env_info.json` and `run_log.txt`

All evaluation policies use common random-number episode seeds. The primary policy is explicitly DQN, or Double DQN when that is the available learned policy. The result is plain: DQN beats Fixed Reorder and EOQ, but trails the tuned `(s,S)` policy.

| Policy | Mean profit per episode (₹) |
| --- | ---: |
| Fixed Reorder | 164,517 |
| EOQ | 221,347 |
| DQN | 234,276 |
| Tuned `(s,S)` | 240,488 |

## Project Layout

`train.py` contains the multi-seed learner, `evaluate.py` runs paired policy evaluation, `analyze.py` produces tables and tests, `plots.py` creates publication figures, and `experiments.py` runs ablations, sensitivities, and non-stationary transfer tests. `index.html` is an illustrative browser demo, not a source of paper results. Files in `legacy/` are superseded prototypes.

## License

MIT. See [LICENSE](LICENSE).

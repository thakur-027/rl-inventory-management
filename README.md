# Deep Reinforcement Learning for Multi-Period Inventory Restocking Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A rigorous, reproducible benchmark and Deep Reinforcement Learning framework for multi-period single-item inventory control under stochastic demand, non-zero lead times, and capacity constraints.

---

## 📌 Executive Summary & Architecture

This repository provides an end-to-end framework resolving key methodological questions in algorithmic inventory replenishment:
1. **Pipeline-Aware State Representation**: Formulates an $(L+1)$-dimensional MDP state vector $\mathbf{s}_t = [I_t/C, O_{1,t}/C, \dots, O_{L,t}/C]$ incorporating orders currently in transit across lead time $L=3$.
2. **Comprehensive Economic Reward**: Eliminates double-counting and missing cost components by enforcing a complete profit formulation:
   $$r_t = \text{Revenue}_t - \text{Procurement}_t - \text{Holding}_t - \text{Stockout}_t - \text{Fixed Order Cost}_t$$
3. **Rigorous Baselines**: Compares DQN against classical inventory theory:
   - **Fixed Reorder Policy**: Orders fixed batch when on-hand inventory drops below threshold.
   - **Pipeline-Aware EOQ Policy**: Evaluates inventory position $IP = I + \sum O_i$ against $s = L\lambda + z\sqrt{L\lambda}$ with order quantity $Q^* = \sqrt{2\lambda K / h}$.
   - **Tuned $(s, S)$ Policy**: Grid-searched on independent training sequences.
   - **Dynamic Programming Benchmark**: Value iteration over discretised inventory states.
4. **Common Random Numbers (CRN)**: All policies are evaluated on identical stochastic demand realisations across all evaluation episodes to enable valid paired hypothesis testing (Paired $t$-test, Wilcoxon signed-rank test).
5. **Zero Heavy Framework Overhead**: Built with pure NumPy neural networks and vectorized operations for deterministic CPU reproducibility and instant execution across any operating system.

---

## 📐 Mathematical Formulation

### 1. State Space ($\mathcal{S}$)
At start of day $t$, the system observation is:
$$\mathbf{s}_t = \left[ \frac{I_t}{C}, \frac{O_{1,t}}{C}, \frac{O_{2,t}}{C}, \dots, \frac{O_{L,t}}{C} \right]^\top \in [0, 1]^{L+1}$$
where $C = 100$ is maximum warehouse capacity, $I_t$ is on-hand inventory after order delivery, and $O_{k,t}$ represents pending orders arriving in $k$ days.

### 2. Action Space ($\mathcal{A}$)
Discrete replenishment quantities snapped to:
$$\mathcal{A} = \{0, 10, 20, 30, 40, 50\} \quad (\text{units})$$

### 3. Step Transition & Timing Convention
Within each discrete step $t \in \{1, \dots, 90\}$:
1. **Delivery Arrival**: Order placed $L$ days ago arrives:
   $$I_t \leftarrow \min\left(I_{t-1} + O_{1, t-1},\; C\right)$$
2. **Order Placement**: Agent selects action $a_t \in \mathcal{A}$. New order enters pipeline: $O_{L, t} = a_t$.
3. **Procurement & Order Incurrence**:
   $$C_{\text{proc}, t} = a_t \times c, \quad C_{\text{order}, t} = K \cdot \mathbb{I}(a_t > 0)$$
   where unit cost $c = \text{₹}250$ and fixed order cost $K = \text{₹}50$.
4. **Demand Realization**: Stochastic daily customer demand $D_t \sim \text{Poisson}(\lambda)$ is realized.
5. **Demand Fulfillment**:
   $$\text{Sales}_t = \min(I_t, D_t), \quad \text{Unmet}_t = D_t - \text{Sales}_t$$
6. **Inventory Update & Holding Cost**: End-of-day inventory is:
   $$I_{t, \text{end}} = I_t - \text{Sales}_t$$
   Holding cost levied on end-of-day stock: $C_{\text{hold}, t} = I_{t, \text{end}} \times h$ ($h = \text{₹}2.50$/unit/day).
7. **Stockout Penalty**: $C_{\text{stock}, t} = \text{Unmet}_t \times p_{\text{stock}}$ ($p_{\text{stock}} = \text{₹}100$/unit).
8. **Reward**:
   $$r_t = (p \times \text{Sales}_t) - C_{\text{proc}, t} - C_{\text{hold}, t} - C_{\text{stock}, t} - C_{\text{order}, t}$$
   with selling price $p = \text{₹}400$.

### 4. Unbiased Service Level Definition
$$\text{Customer Service Level (CSL)} = 1 - \frac{\sum_{t=1}^T \text{Unmet}_t}{\sum_{t=1}^T D_t}$$

---

## ⚙️ Configuration Reference (`config.py`)

All parameters are centrally governed:

| Parameter | Symbol | Paper Value | Description |
|---|---|---|---|
| **Unit Selling Price** | $p$ | ₹400.00 | Revenue earned per unit sold |
| **Unit Procurement Cost** | $c$ | ₹250.00 | Cost per unit ordered (paid at order time) |
| **Holding Cost** | $h$ | ₹2.50 | Per unit held in warehouse at end of day |
| **Stockout Cost** | $p_{\text{stock}}$ | ₹100.00 | Penalty per unit of unmet customer demand |
| **Fixed Order Cost** | $K$ | ₹50.00 | Fixed transaction cost whenever $a_t > 0$ |
| **Mean Daily Demand** | $\lambda$ | 20.0 | Poisson arrival parameter |
| **Lead Time** | $L$ | 3 | Replenishment delivery delay (days) |
| **Warehouse Capacity** | $C$ | 100 | Maximum on-hand storage capacity |
| **Episode Horizon** | $T$ | 90 | Planning horizon in days |
| **Initial Inventory** | $I_0$ | $\sim \mathcal{U}[20, 40]$ | Starting inventory distribution |
| **Replay Capacity** | $|\mathcal{D}|$ | 20,000 | Experience replay buffer size |
| **Target Sync Frequency** | $N_{\text{target}}$ | 500 steps | Hard copy from online to target network |
| **Discount Factor** | $\gamma$ | 0.95 | Future reward discount factor |
| **Exploration Schedule** | $\epsilon$ | $1.0 \to 0.01$ | Linear decay over 36,000 environment steps |

---

## 🚀 Reproduction & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/thakur-027/rl-inventory-management.git
cd rl-inventory-management
pip install -r requirements.txt
```

---

## 📂 Repository Structure

```text
rl-inventory-management/
├── assets/
│   └── figures/             # Interactive demo figures & UI screenshots
├── legacy/
│   └── inventory-rl-training.py  # Original monolithic prototype
├── results/                 # Evaluation CSVs, trained weights, vector PDFs & PNGs
│   ├── ablation/
│   ├── nonstationary/
│   └── sensitivity/
├── agent.py                 # Pure NumPy DQN & Double DQN agent
├── analyze.py               # Statistical tests (TOST, Welch, Holm-Bonferroni)
├── baselines.py             # Classical baselines (EOQ, (s,S), DP)
├── config.py                # Environment & training hyperparameter dataclasses
├── env.py                   # Multi-period inventory gym MDP environment
├── evaluate.py              # Common Random Numbers (CRN) evaluation engine
├── experiments.py           # Ablation, sensitivity & non-stationarity suites
├── index.html               # Self-contained interactive web simulation dashboard
├── plots.py                 # Publication-quality vector figure generator
├── reproduce.py             # Single-command end-to-end reproduction runner
├── train.py                 # Multi-seed training pipeline
├── requirements.txt         # Minimal dependency specifications
├── LICENSE                  # MIT License
└── README.md                # Paper documentation & benchmark guide
```

### 2. Single-Command End-to-End Pipeline
Execute the full scientific pipeline (training 5 seeds × 500 episodes, evaluating on 200 common episodes per seed, statistical hypothesis tests, and vector figure exports):
```bash
python reproduce.py
```

For a rapid smoke test (2 seeds × 100 episodes):
```bash
python reproduce.py --quick
```

### 3. Modular Commands
You can also run individual components independently:

```bash
# 1. Train DQN agents across multiple seeds
python train.py --episodes 500 --seeds 42 123 456 789 1024

# 2. Evaluate all policies using Common Random Numbers
python evaluate.py --eval-episodes 200

# 3. Compute statistical significance tests & summary tables
python analyze.py --csv results/evaluation_results.csv

# 4. Generate publication-ready vector figures (.pdf and .png)
python plots.py --results-dir results
```

### 4. Ablation & Sensitivity Suites
```bash
# Run ablation suite (No Target Net, Small Replay, Double DQN, Inventory-Only State)
python experiments.py --ablation

# Run sensitivity study (Demand λ, Lead Time L, Cost ±20%)
python experiments.py --sensitivity
```

---

## 📊 Generated Artifacts & Figures

The pipeline outputs all results directly to `results/`:

- `results/env_info.json`: Hardware specifications, CPU model, OS, threading, and software versions.
- `results/training_log.csv`: Per-episode training loss, reward, epsilon, and execution duration.
- `results/evaluation_results.csv`: Per-episode records of profit, revenue, procurement, holding, stockout, ordering costs, order count, average inventory, and service level.
- `results/summary_statistics.csv`: Mean, 95% Confidence Interval, std dev, median, and IQR across all metrics.
- `results/hypothesis_tests.json`: Paired $t$-statistics, degrees of freedom, Wilcoxon $W$, $p$-values, Cohen's $d_z$ effect sizes, and win rates.
- `results/fig1_training_dynamics.pdf`: Training convergence curve with multi-seed standard deviation band and $\epsilon$ decay.
- `results/fig2_profit_comparison.pdf`: Net profit comparison bar chart with 95% confidence intervals.
- `results/fig3_service_level.pdf`: Customer demand service level without artificial axis truncations.
- `results/fig4_cost_breakdown.pdf`: Stacked economic cost breakdown per policy.
- `results/fig5_inventory_trajectory.pdf`: 90-day daily inventory level trajectory under identical demand sequences.
- `results/fig6_policy_heatmap.pdf`: Learned DQN replenishment policy decision surface ($I$ vs pipeline $\sum O_i$).

---

## 🖥️ Interactive Web Dashboard

An illustrative interactive browser demo is available in `index.html`:
- Open `index.html` in any modern web browser.
- Run interactive simulated training episodes with visual animations of order pipelines and stock fluctuations.
- Note: The web dashboard is an **illustrative educational demo**. All scientific tables, statistics, and publication figures in the paper are derived directly from the automated Python pipeline in `reproduce.py`.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

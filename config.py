"""Central configuration for the RL Inventory Management project.

Every tunable parameter lives here so that the paper, the code, and
the sensitivity runs all draw from a single source of truth.

Paper reference values
----------------------
price ₹400, cost ₹250, holding ₹2.50/unit/day (end-of-day),
stockout ₹100/unit, order cost ₹50, λ = 20, lead time 3,
capacity 100, horizon 90, start inventory 20–40.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# =====================================================================
# A. Environment
# =====================================================================

@dataclass
class EnvConfig:
    """Inventory environment parameters."""

    # ── Pricing & costs (₹) ──────────────────────────────────────────
    unit_price: float = 400.0             # Revenue per unit sold
    unit_cost: float = 250.0              # Procurement cost per unit ordered
    holding_cost_per_unit: float = 2.5    # Per unit held per day (end-of-day)
    stockout_cost_per_unit: float = 100.0 # Per unit of unmet demand
    fixed_order_cost: float = 50.0        # Fixed cost whenever order_qty > 0

    # ── Demand ────────────────────────────────────────────────────────
    demand_mean: float = 20.0             # λ for Poisson demand
    demand_type: str = "stationary"       # stationary | seasonal | trend | regime
    # Seasonal: multiplicative day-of-week factors (Mon … Sun)
    seasonal_factors: Tuple[float, ...] = (0.8, 1.0, 1.0, 1.1, 1.2, 1.3, 0.6)
    # Trend: λ(t) = demand_mean + trend_slope × t
    trend_slope: float = 0.1
    # Regime shift: λ₂ after regime_shift_day
    regime_lam2: float = 30.0
    regime_shift_day: int = 45

    # ── Inventory ─────────────────────────────────────────────────────
    max_inventory: int = 100              # Warehouse capacity  (C)
    max_order_qty: int = 50               # Maximum single-order quantity
    n_actions: int = 6                    # Discrete actions: 0, 10, 20, 30, 40, 50
    lead_time: int = 3                    # Days until order arrives
    horizon: int = 90                     # Episode length (days)
    init_inventory_low: int = 20          # Starting inventory  ∼ U[low, high]
    init_inventory_high: int = 40

    # ── State features ────────────────────────────────────────────────
    include_demand_avg: bool = False      # Append an EMA of demand to state
    include_day_of_week: bool = False     # Append day % 7 / 6.0
    demand_avg_alpha: float = 0.1         # EMA smoothing coefficient

    # ── Derived helpers ───────────────────────────────────────────────

    @property
    def state_size(self) -> int:
        """Dimensionality of the observation vector."""
        size = 1 + self.lead_time          # [I, O₁, …, O_L]
        if self.include_demand_avg:
            size += 1
        if self.include_day_of_week:
            size += 1
        return size

    @property
    def action_values(self) -> List[int]:
        """Order quantities corresponding to each discrete action index."""
        step = self.max_order_qty // (self.n_actions - 1)
        return [i * step for i in range(self.n_actions)]


# =====================================================================
# B. Agent
# =====================================================================

@dataclass
class AgentConfig:
    """DQN agent hyper-parameters.

    Note: These hyperparameters are frozen for the benchmark. If candidate
    configurations are tested, evaluations must use a validation seed set
    (seeds 2,000,000+), logging every tested configuration to tuning_log.csv,
    leaving the final evaluation test seeds (1,000,000+) strictly untouched.
    """

    gamma: float = 0.95                   # Discount factor
    epsilon_start: float = 1.0            # Initial exploration rate
    epsilon_end: float = 0.01             # Final exploration rate
    epsilon_decay_steps: int = 36_000     # Linear ε-decay over this many steps
    learning_rate: float = 0.001          # Adam learning rate
    batch_size: int = 64                  # Mini-batch size
    replay_buffer_size: int = 20_000      # Experience-replay capacity
    target_update_freq: int = 500         # Hard-sync target net every N steps
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 32])
    double_dqn: bool = False              # Toggle Double-DQN action selection
    warmup_steps: int = 500               # Buffer fill before first update
    gradient_clip: float = 10.0           # Max gradient ‖·‖
    huber_delta: float = 1.0              # Huber-loss transition point
    reward_scale: float = 0.001           # Multiply rewards before storing
    mask_pipeline: bool = False           # If true, agent only sees obs[:1]


# =====================================================================
# C. Training / evaluation
# =====================================================================

@dataclass
class TrainConfig:
    """Training-loop and evaluation settings."""

    episodes: int = 500                   # Training episodes per seed
    eval_episodes: int = 200              # Evaluation episodes per policy/seed
    seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 1024]
    )
    results_dir: str = "results"
    save_weights: bool = True
    log_interval: int = 50                # Print every N episodes


# =====================================================================
# D. Baselines
# =====================================================================

@dataclass
class BaselineConfig:
    """Baseline-policy parameters."""

    # Fixed reorder: order fixed_order_action when I < reorder_point
    fixed_reorder_point: int = 30
    fixed_order_action: int = 3           # action index → 30 units

    # EOQ: Q* = √(2·D·K / h),  s = LT·D + z·√(LT·D)
    eoq_service_z: float = 1.65           # ≈ 95 % cycle service level

    # (s, S) grid search tuning parameters
    ss_tune_episodes: int = 100           # Episodes used during tuning

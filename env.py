"""Inventory management environment following the Gymnasium API.

This implementation follows the Gymnasium reset/step API but does not
subclass ``gym.Env``.

Observation
-----------
[I/C, O₁/C, O₂/C, …, O_L/C]   (optionally + demand_avg/C)

All values normalised by warehouse capacity C.

Actions
-------
Discrete index  →  order quantity  {0, 10, 20, 30, 40, 50}.

Reward
------
r_t = revenue − procurement − holding − stockout − fixed_order_cost

    revenue        = sales × unit_price
    procurement    = order_qty × unit_cost          (paid at order time)
    holding        = I_end × holding_cost_per_unit  (end-of-day inventory)
    stockout       = unmet × stockout_cost_per_unit
    fixed_order    = 𝟙[order_qty > 0] × fixed_order_cost

Timing convention (within one call to ``step(action)``)
-------------------------------------------------------
1. Pending order arrives  →  inventory increases  (start of day).
2. Agent places a new order  →  procurement cost incurred.
3. Stochastic demand realised  →  sales = min(I, D).
4. Inventory updated: I_end = I − sales.
5. Holding cost levied on **end-of-day** inventory.

Gymnasium 0.26+ API shape
--------------------
reset(seed)  →  (obs, info)
step(action) →  (obs, reward, terminated, truncated, info)
"""

from __future__ import annotations

import collections
from typing import Optional, Tuple, Dict, Any

import numpy as np

from config import EnvConfig


# =====================================================================
# Pluggable demand generators
# =====================================================================

class DemandGenerator:
    """Abstract base for demand generators.

    Each generator holds its own ``np.random.Generator`` so that
    seeding the environment produces deterministic demand sequences.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def sample(self, day: int) -> int:
        raise NotImplementedError


class StationaryPoisson(DemandGenerator):
    """D_t ~ Poisson(λ)."""

    def __init__(self, lam: float, rng: np.random.Generator):
        super().__init__(rng)
        self.lam = lam

    def sample(self, day: int) -> int:
        return int(self.rng.poisson(self.lam))


class WeeklySeasonality(DemandGenerator):
    """D_t ~ Poisson(λ × factor[day % 7]).

    Default factors (Mon–Sun): 0.8  1.0  1.0  1.1  1.2  1.3  0.6
    """

    def __init__(self, base_lam: float, rng: np.random.Generator,
                 factors: Tuple[float, ...] = (0.8, 1.0, 1.0, 1.1, 1.2, 1.3, 0.6)):
        super().__init__(rng)
        self.base_lam = base_lam
        self.factors = factors

    def sample(self, day: int) -> int:
        lam = max(0.1, self.base_lam * self.factors[day % 7])
        return int(self.rng.poisson(lam))


class TrendDemand(DemandGenerator):
    """D_t ~ Poisson(max(0.1, λ + slope × t))."""

    def __init__(self, base_lam: float, slope: float,
                 rng: np.random.Generator):
        super().__init__(rng)
        self.base_lam = base_lam
        self.slope = slope

    def sample(self, day: int) -> int:
        lam = max(0.1, self.base_lam + self.slope * day)
        return int(self.rng.poisson(lam))


class RegimeShift(DemandGenerator):
    """D_t ~ Poisson(λ₁) for t < shift_day, else Poisson(λ₂)."""

    def __init__(self, lam1: float, lam2: float, shift_day: int,
                 rng: np.random.Generator):
        super().__init__(rng)
        self.lam1 = lam1
        self.lam2 = lam2
        self.shift_day = shift_day

    def sample(self, day: int) -> int:
        lam = self.lam1 if day < self.shift_day else self.lam2
        return int(self.rng.poisson(lam))


def make_demand_generator(cfg: EnvConfig,
                          rng: np.random.Generator) -> DemandGenerator:
    """Factory: build the demand generator specified by ``cfg.demand_type``."""
    if cfg.demand_type == "stationary":
        return StationaryPoisson(cfg.demand_mean, rng)
    if cfg.demand_type == "seasonal":
        return WeeklySeasonality(cfg.demand_mean, rng, cfg.seasonal_factors)
    if cfg.demand_type == "trend":
        return TrendDemand(cfg.demand_mean, cfg.trend_slope, rng)
    if cfg.demand_type == "regime":
        return RegimeShift(cfg.demand_mean, cfg.regime_lam2,
                           cfg.regime_shift_day, rng)
    raise ValueError(f"Unknown demand_type: {cfg.demand_type!r}")


# =====================================================================
# Inventory environment
# =====================================================================

class InventoryEnv:
    """Single-product inventory management environment.

    State
    ~~~~~
    ``[I/C, O₁/C, …, O_L/C]``  where *C* = ``max_inventory``.

    Optionally appends an exponential moving average of demand
    (``include_demand_avg``).
    """

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        self.action_values = self.cfg.action_values

        # These are (re-)initialised on every reset()
        self.rng: np.random.Generator = np.random.default_rng()
        self.demand_gen: Optional[DemandGenerator] = None
        self.inventory: int = 0
        self.pending_orders: collections.deque = collections.deque(
            [0] * self.cfg.lead_time, maxlen=self.cfg.lead_time
        )
        self.day: int = 0
        self.demand_avg: float = self.cfg.demand_mean

        # Episode-level accumulators for service-level calculation
        self.total_demand: int = 0
        self.total_unmet: int = 0

    # -----------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            If given, creates a fresh ``np.random.Generator`` so the
            episode is fully reproducible.

        Returns
        -------
        obs  : ndarray   – normalised observation vector.
        info : dict      – initial step info (all counters zero).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.demand_gen = make_demand_generator(self.cfg, self.rng)

        self.inventory = int(
            self.rng.integers(self.cfg.init_inventory_low,
                              self.cfg.init_inventory_high + 1)
        )
        self.pending_orders = collections.deque(
            [0] * self.cfg.lead_time, maxlen=self.cfg.lead_time
        )
        self.day = 0
        self.demand_avg = float(self.cfg.demand_mean)
        self.total_demand = 0
        self.total_unmet = 0

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "demand": 0, "sales": 0, "unmet_demand": 0,
            "revenue": 0.0, "procurement_cost": 0.0,
            "holding_cost": 0.0, "stockout_cost": 0.0,
            "fixed_order_cost": 0.0, "order_quantity": 0,
            "order_placed": False, "inventory_end": self.inventory,
            "discarded_stock": 0,
            "service_level": 1.0,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one day.

        Returns
        -------
        obs        : ndarray
        reward     : float
        terminated : bool  (always False — no natural terminal state)
        truncated  : bool  (True when ``day >= horizon``)
        info       : dict  with per-step cost components
        """
        assert 0 <= action < self.cfg.n_actions, f"Invalid action {action}"

        # 1. Receive pending order (start of day)
        arrived = self.pending_orders.popleft()
        new_inv = self.inventory + arrived
        discarded = max(0, new_inv - self.cfg.max_inventory)
        self.inventory = min(new_inv, self.cfg.max_inventory)

        # 2. Place new order  →  procurement & fixed cost
        order_qty = self.action_values[action]
        self.pending_orders.append(order_qty)
        procurement = order_qty * self.cfg.unit_cost
        order_cost = self.cfg.fixed_order_cost if order_qty > 0 else 0.0

        # 3. Realise stochastic demand
        demand = self.demand_gen.sample(self.day)

        # 4. Sales and stock-outs
        sales = min(self.inventory, demand)
        unmet = demand - sales

        # 5. Update inventory  (end of day)
        self.inventory -= sales

        # 6. Cost components
        revenue = sales * self.cfg.unit_price
        holding = self.inventory * self.cfg.holding_cost_per_unit
        stockout = unmet * self.cfg.stockout_cost_per_unit

        # 7. Reward
        reward = revenue - procurement - holding - stockout - order_cost

        # 8. Book-keeping
        self.day += 1
        self.total_demand += demand
        self.total_unmet += unmet
        if self.cfg.include_demand_avg:
            a = self.cfg.demand_avg_alpha
            self.demand_avg = (1 - a) * self.demand_avg + a * demand

        # 9. Termination signals
        terminated = False
        truncated = self.day >= self.cfg.horizon

        # 10. Service level (cumulative over the episode)
        service_level = (
            1.0 - self.total_unmet / self.total_demand
            if self.total_demand > 0 else 1.0
        )

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "demand": demand,
            "sales": sales,
            "unmet_demand": unmet,
            "revenue": revenue,
            "procurement_cost": procurement,
            "holding_cost": holding,
            "stockout_cost": stockout,
            "fixed_order_cost": order_cost,
            "order_quantity": order_qty,
            "order_placed": order_qty > 0,
            "inventory_end": self.inventory,
            "discarded_stock": discarded,
            "service_level": service_level,
        }
        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the normalised observation vector."""
        cap = float(self.cfg.max_inventory)
        parts = [self.inventory / cap]
        parts.extend(o / cap for o in self.pending_orders)
        if self.cfg.include_demand_avg:
            parts.append(self.demand_avg / cap)
        if self.cfg.include_day_of_week:
            parts.append((self.day % 7) / 6.0)
        return np.array(parts, dtype=np.float32)

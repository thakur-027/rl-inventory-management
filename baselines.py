"""Baseline inventory control policies: Fixed Reorder, EOQ, and (s, S) Grid Search.

All policies adhere to a common interface:
    `act(obs: np.ndarray) -> int`  (returns discrete action index)
and optionally:
    `reset() -> None`
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from config import BaselineConfig, EnvConfig


# =====================================================================
# Helper: Action matching
# =====================================================================

def closest_action_index(target_qty: float, action_values: List[int]) -> int:
    """Find the discrete action index whose order quantity is closest to target_qty."""
    diffs = [abs(target_qty - qty) for qty in action_values]
    return int(np.argmin(diffs))


# =====================================================================
# Base Policy Interface
# =====================================================================

class BasePolicy:
    """Abstract interface for all inventory policies."""

    def __init__(self, env_cfg: EnvConfig):
        self.env_cfg = env_cfg
        self.action_values = env_cfg.action_values
        self.capacity = float(env_cfg.max_inventory)

    def reset(self) -> None:
        """Reset internal state if applicable."""
        pass

    def act(self, obs: np.ndarray) -> int:
        """Select action index given normalized observation."""
        raise NotImplementedError


# =====================================================================
# 1. Fixed Reorder Policy
# =====================================================================

class FixedReorderPolicy(BasePolicy):
    """Fixed Reorder Policy: If on-hand inventory < reorder_point, order fixed quantity.

    Supports paper specification: order 30 units when I < 30 (or custom configured).
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        base_cfg: Optional[BaselineConfig] = None,
        reorder_point: Optional[int] = None,
        order_action: Optional[int] = None,
    ):
        super().__init__(env_cfg)
        cfg_base = base_cfg if base_cfg is not None else BaselineConfig()
        self.reorder_point = (
            reorder_point if reorder_point is not None else cfg_base.fixed_reorder_point
        )
        self.order_action = (
            order_action if order_action is not None else cfg_base.fixed_order_action
        )

    def act(self, obs: np.ndarray) -> int:
        # obs[0] is normalized inventory I / C
        on_hand = obs[0] * self.capacity
        if on_hand < self.reorder_point:
            return self.order_action
        return 0


# =====================================================================
# 2. Economic Order Quantity (EOQ) Policy with Pipeline Awareness
# =====================================================================

class EOQPolicy(BasePolicy):
    """EOQ-based (s, Q) replenishment policy.

    Formulation:
    -----------
    Economic Order Quantity:
        Q* = sqrt(2 * D * K / h)
    where:
        D = daily demand mean (cfg.demand_mean)
        K = fixed ordering cost (cfg.fixed_order_cost)
        h = holding cost per unit per day (cfg.holding_cost_per_unit)

    Reorder Point (s):
        s = lead_time_demand + safety_stock
        lead_time_demand = L * D
        safety_stock = z * sqrt(L * D)  (for Poisson demand variance = mean)

    Control Rule:
    ------------
    Evaluated on Inventory Position (IP = on-hand + on-order):
        If IP <= s: order Q* (snapped to closest discrete action)
        Else: order 0
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        service_z: float = 1.65,
    ):
        super().__init__(env_cfg)
        self.service_z = service_z

        d = env_cfg.demand_mean
        k = env_cfg.fixed_order_cost
        h = env_cfg.holding_cost_per_unit
        lt = env_cfg.lead_time

        # Calculate theoretical EOQ
        if h > 0:
            self.eoq_q = np.sqrt(2.0 * d * k / h)
        else:
            self.eoq_q = float(env_cfg.max_order_qty)

        # Calculate reorder point s
        lt_mean = lt * d
        lt_std = np.sqrt(lt_mean)
        self.reorder_point = lt_mean + self.service_z * lt_std

        # Action index corresponding to Q*
        self.order_action = closest_action_index(self.eoq_q, self.action_values)
        if self.order_action == 0:
            self.order_action = 1  # ensure at least smallest nonzero order if triggered

    def act(self, obs: np.ndarray) -> int:
        # Calculate inventory position IP = on_hand + sum(pipeline orders)
        # obs = [I/C, O1/C, ..., OL/C]
        num_orders = self.env_cfg.lead_time
        inv_position = (obs[0] + np.sum(obs[1 : 1 + num_orders])) * self.capacity

        if inv_position <= self.reorder_point:
            return self.order_action
        return 0


# =====================================================================
# 3. (s, S) Policy & Adaptive Grid Search Tuner
# =====================================================================

class SSPolicy(BasePolicy):
    """(s, S) Order-Up-To Policy using Inventory Position.

    When IP <= s, place an order to bring IP as close to S as possible
    within available discrete action quantities.
    """

    def __init__(self, env_cfg: EnvConfig, s: float, S: float):
        super().__init__(env_cfg)
        self.s = s
        self.S = S

    def act(self, obs: np.ndarray) -> int:
        num_orders = self.env_cfg.lead_time
        inv_position = (obs[0] + np.sum(obs[1 : 1 + num_orders])) * self.capacity

        if inv_position <= self.s:
            desired_order = max(0.0, self.S - inv_position)
            # Find best discrete action that gets closest to desired_order
            return closest_action_index(desired_order, self.action_values)
        return 0


def tune_ss_policy(
    env_cfg: EnvConfig,
    base_cfg: Optional[BaselineConfig] = None,
    tune_episodes: int = 100,
    seed: int = 9999,
) -> Tuple[SSPolicy, float, Tuple[int, int]]:
    """Grid-search the optimal (s, S) policy on independent training demand sequences.

    Uses an adaptive grid anchored to expected lead-time demand (L · λ) to guarantee
    sufficient coverage for different lead times (e.g. L=5) and demand rates (λ=25).
    Issues a warning if the tuned parameters land on a grid boundary.
    """
    from env import InventoryEnv

    base_cfg = base_cfg or BaselineConfig()

    # Adaptive search grid anchored to lead-time demand (L · λ)
    lt_demand = float(env_cfg.lead_time * env_cfg.demand_mean)
    s_min = max(0, int(math.floor(0.5 * lt_demand / 5.0) * 5))
    s_max = max(s_min + 5, int(math.ceil(2.5 * lt_demand / 5.0) * 5))
    s_vals = list(range(s_min, s_max + 1, 5))

    S_max = int(math.ceil((s_max + env_cfg.max_order_qty + 10) / 5.0) * 5)
    S_vals = list(range(s_min + 5, S_max + 1, 5))

    # Pre-generate tuning seeds to ensure all candidate pairs see identical environments
    eval_seeds = [seed + i for i in range(tune_episodes)]

    best_profit = -float("inf")
    best_params = (s_vals[0], S_vals[0])

    # Instantiate single environment
    test_env = InventoryEnv(env_cfg)

    for s, S in itertools.product(s_vals, S_vals):
        if s >= S:
            continue

        policy = SSPolicy(env_cfg, s=s, S=S)
        total_profit = 0.0

        for ep_seed in eval_seeds:
            obs, _ = test_env.reset(seed=ep_seed)
            ep_reward = 0.0
            done = False
            while not done:
                action = policy.act(obs)
                obs, reward, term, trunc, _ = test_env.step(action)
                ep_reward += reward
                done = term or trunc

            total_profit += ep_reward

        avg_profit = total_profit / tune_episodes
        if avg_profit > best_profit:
            best_profit = avg_profit
            best_params = (s, S)

    # Check for boundary landing
    on_s_edge = (best_params[0] == s_vals[0] or best_params[0] == s_vals[-1])
    on_S_edge = (best_params[1] == S_vals[0] or best_params[1] == S_vals[-1])
    if on_s_edge or on_S_edge:
        warnings.warn(
            f"Tuned (s, S) {best_params} landed on grid boundary: "
            f"s in [{s_vals[0]}, {s_vals[-1]}], S in [{S_vals[0]}, {S_vals[-1]}]."
        )

    best_policy = SSPolicy(env_cfg, s=best_params[0], S=best_params[1])
    return best_policy, best_profit, best_params


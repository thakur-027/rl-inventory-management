"""Baseline inventory control policies: Fixed Reorder, EOQ, (s, S) Grid Search, and DP.

All policies adhere to a common interface:
    `act(obs: np.ndarray) -> int`  (returns discrete action index)
and optionally:
    `reset() -> None`
"""

from __future__ import annotations

import itertools
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
        reorder_point: Optional[int] = None,
        order_action: Optional[int] = None,
    ):
        super().__init__(env_cfg)
        cfg_base = BaselineConfig()
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
# 3. (s, S) Policy & Grid Search Tuner
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

    Parameters
    ----------
    env_cfg : EnvConfig
    base_cfg : BaselineConfig
    tune_episodes : number of evaluation episodes to average over for each candidate pair
    seed : seed for reproducible tuning sequences

    Returns
    -------
    best_policy : SSPolicy
    best_avg_profit : float
    best_params : (s, S)
    """
    from env import InventoryEnv

    base_cfg = base_cfg or BaselineConfig()
    s_vals = base_cfg.ss_s_values
    S_vals = base_cfg.ss_S_values

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

    best_policy = SSPolicy(env_cfg, s=best_params[0], S=best_params[1])
    return best_policy, best_profit, best_params


# =====================================================================
# 4. Discrete Dynamic Programming / Value Iteration Benchmark (Optional)
# =====================================================================

class DPValueIterationPolicy(BasePolicy):
    """Dynamic programming value iteration benchmark on inventory position MDP.

    Under stationary demand and constant lead time, the system can be modeled
    using the inventory position state IP, yielding an optimal control policy.
    """

    def __init__(self, env_cfg: EnvConfig, base_cfg: Optional[BaselineConfig] = None):
        super().__init__(env_cfg)
        self.base_cfg = base_cfg or BaselineConfig()
        self.policy_table: Dict[int, int] = {}
        self._solve_value_iteration()

    def _solve_value_iteration(
        self, gamma: float = 0.95, max_iter: int = 100, tol: float = 1e-3
    ) -> None:
        """Solve value iteration over discretised inventory positions."""
        step = self.base_cfg.dp_inventory_step
        max_ip = self.env_cfg.max_inventory + self.env_cfg.max_order_qty
        states = list(range(0, max_ip + 1, step))
        n_states = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}

        V = np.zeros(n_states)

        # Precompute Poisson PMF up to dp_max_demand
        max_d = self.base_cfg.dp_max_demand
        lam = self.env_cfg.demand_mean
        d_vals = np.arange(0, max_d + 1)
        # Poisson PMF
        from scipy.stats import poisson
        pmf = poisson.pmf(d_vals, lam)
        pmf /= pmf.sum()  # normalize

        # Value iteration loop
        for _ in range(max_iter):
            delta = 0.0
            new_V = np.zeros(n_states)
            for i, ip in enumerate(states):
                best_val = -float("inf")
                for a_idx, qty in enumerate(self.action_values):
                    # Check capacity
                    if ip + qty > max_ip:
                        continue

                    # Expected immediate reward
                    order_cost = (
                        (qty * self.env_cfg.unit_cost + self.env_cfg.fixed_order_cost)
                        if qty > 0
                        else 0.0
                    )

                    ev = 0.0
                    for d, prob in zip(d_vals, pmf):
                        sales = min(ip, d)
                        unmet = d - sales
                        next_ip = max(0, ip + qty - d)

                        rev = sales * self.env_cfg.unit_price
                        holding = next_ip * self.env_cfg.holding_cost_per_unit
                        stockout = unmet * self.env_cfg.stockout_cost_per_unit
                        reward = rev - order_cost - holding - stockout

                        # Snap next_ip to nearest grid state
                        closest_s = min(states, key=lambda s: abs(s - next_ip))
                        j = state_to_idx[closest_s]
                        ev += prob * (reward + gamma * V[j])

                    if ev > best_val:
                        best_val = ev

                new_V[i] = best_val
                delta = max(delta, abs(new_V[i] - V[i]))

            V = new_V
            if delta < tol:
                break

        # Extract greedy policy
        for i, ip in enumerate(states):
            best_val = -float("inf")
            best_act = 0
            for a_idx, qty in enumerate(self.action_values):
                if ip + qty > max_ip:
                    continue
                order_cost = (
                    (qty * self.env_cfg.unit_cost + self.env_cfg.fixed_order_cost)
                    if qty > 0
                    else 0.0
                )
                ev = 0.0
                for d, prob in zip(d_vals, pmf):
                    sales = min(ip, d)
                    unmet = d - sales
                    next_ip = max(0, ip + qty - d)
                    rev = sales * self.env_cfg.unit_price
                    holding = next_ip * self.env_cfg.holding_cost_per_unit
                    stockout = unmet * self.env_cfg.stockout_cost_per_unit
                    reward = rev - order_cost - holding - stockout
                    closest_s = min(states, key=lambda s: abs(s - next_ip))
                    j = state_to_idx[closest_s]
                    ev += prob * (reward + gamma * V[j])

                if ev > best_val:
                    best_val = ev
                    best_act = a_idx
            self.policy_table[ip] = best_act

    def act(self, obs: np.ndarray) -> int:
        num_orders = self.env_cfg.lead_time
        inv_position = int(round((obs[0] + np.sum(obs[1 : 1 + num_orders])) * self.capacity))
        # Find closest grid state in policy table
        closest_state = min(self.policy_table.keys(), key=lambda s: abs(s - inv_position))
        return self.policy_table[closest_state]

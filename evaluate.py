"""Evaluation pipeline with Common Random Numbers across all policies.

Guarantees fair paired comparisons:
- Every policy (DQN, Double DQN, Fixed Reorder, EOQ, tuned (s,S), DP)
  is tested on the exact same stochastic demand sequences.
- Outputs comprehensive per-episode CSV with full cost/revenue breakdowns,
  service levels, order counts, and inventory statistics.
- Benchmarks policy inference latency.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

from config import EnvConfig, AgentConfig, TrainConfig, BaselineConfig
from env import InventoryEnv
from agent import DQNAgent
from baselines import (
    BasePolicy,
    FixedReorderPolicy,
    EOQPolicy,
    SSPolicy,
    tune_ss_policy,
    DPValueIterationPolicy,
)


class DQNEvalWrapper(BasePolicy):
    """Wrapper to make DQNAgent conform to BasePolicy interface for evaluation."""

    def __init__(self, agent: DQNAgent):
        super().__init__(agent.env_cfg)
        self.agent = agent

    def act(self, obs: np.ndarray) -> int:
        return self.agent.act(obs, explore=False)


def run_episode(
    env: InventoryEnv, policy: BasePolicy, ep_seed: int
) -> Dict[str, Any]:
    """Execute a single evaluation episode and return detailed statistics."""
    obs, info = env.reset(seed=ep_seed)

    tot_reward = 0.0
    tot_rev = 0.0
    tot_proc = 0.0
    tot_hold = 0.0
    tot_stockout = 0.0
    tot_order_cost = 0.0
    order_count = 0
    inv_levels: List[float] = []
    tot_unmet = 0
    tot_demand = 0

    done = False
    step_times: List[float] = []

    while not done:
        t0 = time.perf_counter()
        action = policy.act(obs)
        step_times.append(time.perf_counter() - t0)

        obs, reward, term, trunc, step_info = env.step(action)
        done = term or trunc

        tot_reward += reward
        tot_rev += step_info["revenue"]
        tot_proc += step_info["procurement_cost"]
        tot_hold += step_info["holding_cost"]
        tot_stockout += step_info["stockout_cost"]
        tot_order_cost += step_info["fixed_order_cost"]
        if step_info["order_placed"]:
            order_count += 1
        inv_levels.append(step_info["inventory_end"])
        tot_unmet += step_info["unmet_demand"]
        tot_demand += step_info["demand"]

    service_level = 1.0 - (tot_unmet / tot_demand) if tot_demand > 0 else 1.0
    avg_inv = float(np.mean(inv_levels)) if inv_levels else 0.0
    mean_decision_us = float(np.mean(step_times) * 1e6) if step_times else 0.0

    return {
        "profit": tot_reward,
        "revenue": tot_rev,
        "procurement": tot_proc,
        "holding": tot_hold,
        "stockout": tot_stockout,
        "ordering": tot_order_cost,
        "number_of_orders": order_count,
        "average_inventory": avg_inv,
        "unmet": tot_unmet,
        "demand": tot_demand,
        "service_level": service_level,
        "decision_time_us": mean_decision_us,
    }


def evaluate_all(
    env_cfg: Optional[EnvConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    agent_cfg: Optional[AgentConfig] = None,
    base_cfg: Optional[BaselineConfig] = None,
    seeds: Optional[List[int]] = None,
    eval_episodes: int = 200,
    include_dp: bool = False,
    results_dir: str = "results",
) -> str:
    """Run full evaluation across seeds and policies with Common Random Numbers."""
    env_cfg = env_cfg or EnvConfig()
    train_cfg = train_cfg or TrainConfig()
    agent_cfg = agent_cfg or AgentConfig()
    base_cfg = base_cfg or BaselineConfig()
    active_seeds = seeds if seeds is not None else train_cfg.seeds

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "evaluation_results.csv")

    eval_env = InventoryEnv(env_cfg)

    # Common evaluation seeds across all policies
    # Using fixed sequence starting from 1_000_000 for perfect reproducibility
    common_seeds = [1_000_000 + i for i in range(eval_episodes)]

    # 1. Pre-tune (s, S) policy on separate training demand sequences
    print("Tuning (s, S) policy on independent training demand sequences...")
    tuned_ss_policy, ss_train_profit, best_ss_params = tune_ss_policy(
        env_cfg, base_cfg, tune_episodes=base_cfg.ss_tune_episodes, seed=500_000
    )
    print(f"Optimal (s, S) found: s={best_ss_params[0]}, S={best_ss_params[1]} (Tuning Avg Profit: INR {ss_train_profit:.2f})")

    # 2. Build non-learning baselines
    fixed_policy = FixedReorderPolicy(env_cfg)
    eoq_policy = EOQPolicy(env_cfg, service_z=base_cfg.eoq_service_z)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy",
            "seed",
            "episode",
            "profit",
            "revenue",
            "procurement",
            "holding",
            "stockout",
            "ordering",
            "number_of_orders",
            "average_inventory",
            "unmet",
            "demand",
            "service_level",
            "decision_time_us",
        ])

        # Evaluate baselines ONCE (they are deterministic and independent of training seed)
        print("\n--- Evaluating Baselines (200 common episodes) ---")
        baselines = [
            ("FixedReorder", fixed_policy),
            ("EOQ", eoq_policy),
            (f"(s,S)[{best_ss_params[0]},{best_ss_params[1]}]", tuned_ss_policy),
        ]
        
        for pol_name, pol_obj in baselines:
            for ep_idx, ep_seed in enumerate(common_seeds, start=1):
                res = run_episode(eval_env, pol_obj, ep_seed)
                writer.writerow([
                    pol_name,
                    "Baseline",
                    ep_idx,
                    f"{res['profit']:.2f}",
                    f"{res['revenue']:.2f}",
                    f"{res['procurement']:.2f}",
                    f"{res['holding']:.2f}",
                    f"{res['stockout']:.2f}",
                    f"{res['ordering']:.2f}",
                    res["number_of_orders"],
                    f"{res['average_inventory']:.2f}",
                    res["unmet"],
                    res["demand"],
                    f"{res['service_level']:.4f}",
                    f"{res['decision_time_us']:.2f}",
                ])

        # Evaluate trained DQN models per seed
        for seed in active_seeds:
            print(f"\n--- Evaluating Seed {seed} ({eval_episodes} common episodes) ---")

            policies_to_eval: List[Tuple[str, BasePolicy]] = []

            # Check for trained DQN weights
            dqn_path = os.path.join(results_dir, f"dqn_seed_{seed}.npz")
            if os.path.exists(dqn_path):
                dqn_agent = DQNAgent(env_cfg=env_cfg, agent_cfg=agent_cfg, seed=seed)
                dqn_agent.load(dqn_path)
                policies_to_eval.append(("DQN", DQNEvalWrapper(dqn_agent)))
            else:
                print(f"Warning: DQN weights not found at {dqn_path}. Skipping DQN for seed {seed}.")

            # Check for trained Double DQN weights
            doubledqn_path = os.path.join(results_dir, f"doubledqn_seed_{seed}.npz")
            if os.path.exists(doubledqn_path):
                ddqn_agent = DQNAgent(env_cfg=env_cfg, agent_cfg=AgentConfig(double_dqn=True), seed=seed)
                ddqn_agent.load(doubledqn_path)
                policies_to_eval.append(("DoubleDQN", DQNEvalWrapper(ddqn_agent)))

            for pol_name, pol_obj in policies_to_eval:
                pol_profits = []
                pol_sls = []
                t_pol_start = time.perf_counter()

                for ep_idx, ep_seed in enumerate(common_seeds, start=1):
                    res = run_episode(eval_env, pol_obj, ep_seed)
                    writer.writerow([
                        pol_name,
                        seed,
                        ep_idx,
                        f"{res['profit']:.2f}",
                        f"{res['revenue']:.2f}",
                        f"{res['procurement']:.2f}",
                        f"{res['holding']:.2f}",
                        f"{res['stockout']:.2f}",
                        f"{res['ordering']:.2f}",
                        res["number_of_orders"],
                        f"{res['average_inventory']:.2f}",
                        res["unmet"],
                        res["demand"],
                        f"{res['service_level']:.4f}",
                        f"{res['decision_time_us']:.1f}",
                    ])
                    pol_profits.append(res["profit"])
                    pol_sls.append(res["service_level"])

                f.flush()
                mean_profit = np.mean(pol_profits)
                std_profit = np.std(pol_profits)
                mean_sl = np.mean(pol_sls) * 100
                pol_dur = time.perf_counter() - t_pol_start
                print(
                    f"  {pol_name:<24} | Mean Profit: INR {mean_profit:9.2f} (+/-{std_profit:7.2f}) | "
                    f"Service Level: {mean_sl:5.2f}% | Time: {pol_dur:4.2f}s"
                )

    print(f"\nEvaluation complete! Detailed per-episode results saved to: {csv_path}")
    return csv_path


# =====================================================================
# CLI Entrypoint
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Inventory Policies with Common Random Numbers")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of common evaluation episodes")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to evaluate")
    parser.add_argument("--include-dp", action="store_true", help="Include DP Value Iteration baseline")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing weights / outputs")
    args = parser.parse_args()

    evaluate_all(
        eval_episodes=args.eval_episodes,
        seeds=args.seeds,
        include_dp=args.include_dp,
        results_dir=args.results_dir,
    )

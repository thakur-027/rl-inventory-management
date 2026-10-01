"""Ablation and sensitivity experiment suite for RL Inventory Optimization.

Provides structured execution for:
1. Ablations:
   - Full DQN (Pipeline state, Target Net, 20k Replay)
   - No Target Network
   - Inventory-Only State (no pipeline information)
   - Small Replay Buffer (500 transitions)
   - Double DQN
2. Sensitivity Analyses:
   - Demand Mean λ ∈ {15, 20, 25}
   - Lead Time L ∈ {3, 5}
   - Cost Multipliers (±20% on Procurement, Holding, Stockout)
3. Non-Stationary Demand Generalization:
   - Stationary Poisson
   - Weekly Seasonality
   - Upward Trend
   - Mid-Horizon Regime Shift
"""

from __future__ import annotations

import argparse
import csv
import os
import json
from typing import Dict, List, Any

# Set BLAS thread limits before importing NumPy (and modules that import it).
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

from config import EnvConfig, AgentConfig, TrainConfig
from train import train_seed
from evaluate import evaluate_all
from analyze import analyze_results


# =====================================================================
# 1. Ablation Suite
# =====================================================================

def run_ablation_study(
    results_dir: str = "results/ablation",
    episodes: int = 500,
    eval_episodes: int = 200,
    seeds: List[int] = [42, 123, 456, 789, 1024],
) -> None:
    """Run ablation suite across core algorithmic components."""
    os.makedirs(results_dir, exist_ok=True)
    print("\n" + "=" * 80)
    print("STARTING ABLATION SUITE")
    print("=" * 80)

    ablation_configs = {
        "Full_DQN": (EnvConfig(), AgentConfig()),
        "No_Target_Net": (EnvConfig(), AgentConfig(target_update_freq=1)),
        "Small_Replay_Buffer": (EnvConfig(), AgentConfig(replay_buffer_size=500)),
        "Double_DQN": (EnvConfig(), AgentConfig(double_dqn=True)),
    }

    # Also Inventory-Only state (lead time = 0 in state representation)
    inv_only_agent_cfg = AgentConfig(mask_pipeline=True)
    ablation_configs["Inv_Only_State"] = (EnvConfig(), inv_only_agent_cfg)

    ablation_results = []

    for name, (e_cfg, a_cfg) in ablation_configs.items():
        print(f"\n>>> Running Ablation Variant: {name} <<<")
        sub_dir = os.path.join(results_dir, name)
        os.makedirs(sub_dir, exist_ok=True)

        t_cfg = TrainConfig(episodes=episodes, results_dir=sub_dir)

        # Train across seeds
        log_csv = os.path.join(sub_dir, "training_log.csv")
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["policy", "seed", "episode", "total_steps", "reward", "epsilon", "loss", "duration_sec"])
            for s in seeds:
                train_seed(s, e_cfg, a_cfg, t_cfg, writer, f)

        # Evaluate
        eval_csv = evaluate_all(e_cfg, t_cfg, agent_cfg=a_cfg, seeds=seeds, eval_episodes=eval_episodes, results_dir=sub_dir)
        # Summarize
        analyze_results(eval_csv, sub_dir)

        # Extract DQN summary
        summary_csv = os.path.join(sub_dir, "summary_statistics.csv")
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["policy"] in ("DQN", "DoubleDQN"):
                    ablation_results.append({
                        "Variant": name,
                        "Mean_Profit": float(row["profit_mean"]),
                        "Std_Profit": float(row["profit_std"]),
                        "Service_Level": float(row["service_level_mean"]),
                        "Avg_Inventory": float(row["avg_inventory_mean"]),
                    })
                    break

    # Save comparative table
    out_table = os.path.join(results_dir, "ablation_summary.json")
    with open(out_table, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETED")
    print(f"{'Variant':<25} | {'Mean Profit (₹)':<16} | {'Service Level':<14} | {'Avg Inv':<8}")
    print("-" * 80)
    for res in ablation_results:
        print(f"{res['Variant']:<25} | ₹{res['Mean_Profit']:14.2f} | {res['Service_Level']:12.2f}% | {res['Avg_Inventory']:8.1f}")
    print("=" * 80)


# =====================================================================
# 2. Sensitivity Suite
# =====================================================================

def run_sensitivity_study(
    results_dir: str = "results/sensitivity",
    episodes: int = 500,
    eval_episodes: int = 200,
    seeds: List[int] = [42, 123, 456, 789, 1024],
) -> None:
    """Run sensitivity analysis across demand rates, lead times, and cost structures."""
    os.makedirs(results_dir, exist_ok=True)
    print("\n" + "=" * 80)
    print("STARTING SENSITIVITY SUITE")
    print("=" * 80)

    variations = {
        "Demand_Lam_15": EnvConfig(demand_mean=15.0),
        "Demand_Lam_20_Base": EnvConfig(demand_mean=20.0),
        "Demand_Lam_25": EnvConfig(demand_mean=25.0),
        "LeadTime_3_Base": EnvConfig(lead_time=3),
        "LeadTime_5": EnvConfig(lead_time=5),
        "Cost_Plus_20Pct": EnvConfig(unit_cost=300.0, holding_cost_per_unit=3.0, stockout_cost_per_unit=120.0),
        "Cost_Minus_20Pct": EnvConfig(unit_cost=200.0, holding_cost_per_unit=2.0, stockout_cost_per_unit=80.0),
    }

    sensitivity_results = []
    completed_configs: Dict[str, str] = {}

    for name, e_cfg in variations.items():
        print(f"\n>>> Running Sensitivity Scenario: {name} <<<")
        sub_dir = os.path.join(results_dir, name)
        os.makedirs(sub_dir, exist_ok=True)

        config_key = repr(e_cfg)
        if config_key in completed_configs:
            source_dir = completed_configs[config_key]
            for filename in ("training_log.csv", "evaluation_results.csv", "summary_statistics.csv", "hypothesis_tests.json", "tuned_ss.json"):
                source = os.path.join(source_dir, filename)
                target = os.path.join(sub_dir, filename)
                if os.path.exists(source) and not os.path.exists(target):
                    import shutil
                    shutil.copy2(source, target)
            print(f"Reusing completed run from {source_dir} for identical configuration.")
            _append_summary_rows(sub_dir, name, sensitivity_results)
            continue
        completed_configs[config_key] = sub_dir

        t_cfg = TrainConfig(episodes=episodes, results_dir=sub_dir)
        a_cfg = AgentConfig()

        log_csv = os.path.join(sub_dir, "training_log.csv")
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["policy", "seed", "episode", "total_steps", "reward", "epsilon", "loss", "duration_sec"])
            for s in seeds:
                train_seed(s, e_cfg, a_cfg, t_cfg, writer, f)

        eval_csv = evaluate_all(e_cfg, t_cfg, agent_cfg=a_cfg, seeds=seeds, eval_episodes=eval_episodes, results_dir=sub_dir)
        analyze_results(eval_csv, sub_dir)

        summary_csv = os.path.join(sub_dir, "summary_statistics.csv")
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sensitivity_results.append({
                    "Scenario": name,
                    "Policy": row["policy"],
                    "Mean_Profit": float(row["profit_mean"]),
                    "Service_Level": float(row["service_level_mean"]),
                    "Avg_Inventory": float(row["avg_inventory_mean"]),
                })

    out_table = os.path.join(results_dir, "sensitivity_summary.json")
    with open(out_table, "w", encoding="utf-8") as f:
        json.dump(sensitivity_results, f, indent=2)

    print("\n" + "=" * 90)
    print("SENSITIVITY ANALYSIS COMPLETED")
    print(f"{'Scenario':<25} | {'Policy':<18} | {'Mean Profit (₹)':<16} | {'Service Level':<14} | {'Avg Inv':<8}")
    print("-" * 90)
    for res in sensitivity_results:
        print(f"{res['Scenario']:<25} | {res['Policy']:<18} | ₹{res['Mean_Profit']:14.2f} | {res['Service_Level']:12.2f}% | {res['Avg_Inventory']:8.1f}")
    print("=" * 90)


# =====================================================================
# 3. Non-Stationary Demand Suite
# =====================================================================

def run_nonstationary_study(
    results_dir: str = "results/nonstationary",
    episodes: int = 500,
    eval_episodes: int = 200,
    seeds: List[int] = [42, 123, 456, 789, 1024],
) -> None:
    """Test DQN robustness under non-stationary demand patterns.

    For each demand scenario we produce two results:
    1. **Zero-shot transfer** – the DQN trained on *stationary* demand is
       evaluated on the non-stationary pattern (no re-training).
    2. **Specialised** – a DQN trained directly on the non-stationary
       demand pattern.

    This lets us quantify both the transfer gap and any benefit of
    demand-aware state features (e.g. ``include_day_of_week`` for
    the seasonal variant).
    """
    os.makedirs(results_dir, exist_ok=True)
    print("\n" + "=" * 80)
    print("STARTING NON-STATIONARY DEMAND SUITE")
    print("=" * 80)

    demand_scenarios: Dict[str, EnvConfig] = {
        "Stationary_Base": EnvConfig(demand_type="stationary"),
        "Weekly_Seasonal": EnvConfig(
            demand_type="seasonal",
            include_day_of_week=True,
        ),
        "Upward_Trend": EnvConfig(
            demand_type="trend",
            trend_slope=0.15,          # λ goes from 20→33.5 over 90 days
            include_demand_avg=True,
        ),
        "Regime_Shift_Day45": EnvConfig(
            demand_type="regime",
            regime_lam2=30.0,
            regime_shift_day=45,
            include_demand_avg=True,
        ),
    }

    nonstat_results: List[Dict[str, Any]] = []

    # --- Phase 1: Train on stationary (for zero-shot transfer) ---
    stat_dir = os.path.join(results_dir, "Stationary_Base")
    os.makedirs(stat_dir, exist_ok=True)
    stat_cfg = demand_scenarios["Stationary_Base"]
    a_cfg = AgentConfig()
    t_cfg = TrainConfig(episodes=episodes, results_dir=stat_dir)

    print("\n>>> Phase 1: Training stationary-demand DQN <<<")
    log_csv = os.path.join(stat_dir, "training_log.csv")
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "seed", "episode", "total_steps", "reward",
                         "epsilon", "loss", "duration_sec"])
        for s in seeds:
            train_seed(s, stat_cfg, a_cfg, t_cfg, writer, f)

    # Evaluate stationary agent on stationary demand
    eval_csv = evaluate_all(stat_cfg, t_cfg, agent_cfg=a_cfg,
                            seeds=seeds, eval_episodes=eval_episodes,
                            results_dir=stat_dir)
    analyze_results(eval_csv, stat_dir)
    _collect_summary(stat_dir, "Stationary_Base", "Stationary_Train",
                     nonstat_results)

    # --- Phase 2: Zero-shot transfer to each non-stationary pattern ---
    print("\n>>> Phase 2: Zero-shot transfer evaluation <<<")
    for scenario_name, e_cfg in demand_scenarios.items():
        if scenario_name == "Stationary_Base":
            continue

        zs_dir = os.path.join(results_dir, f"{scenario_name}_ZeroShot")
        os.makedirs(zs_dir, exist_ok=True)

        # Copy the stationary-trained weights into this directory
        for s in seeds:
            src = os.path.join(stat_dir, f"dqn_seed_{s}.npz")
            dst = os.path.join(zs_dir, f"dqn_seed_{s}.npz")
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)

        # If the scenario uses day-of-week, the stationary agent was NOT
        # trained with that feature → we must use the *stationary* state
        # config for the agent so dimensions match.
        zs_agent_cfg = AgentConfig()
        zs_env_for_eval = EnvConfig(
            demand_type=e_cfg.demand_type,
            seasonal_factors=e_cfg.seasonal_factors,
            trend_slope=e_cfg.trend_slope,
            regime_lam2=e_cfg.regime_lam2,
            regime_shift_day=e_cfg.regime_shift_day,
            include_day_of_week=False,   # match the stationary agent's state
        )
        zs_t_cfg = TrainConfig(episodes=episodes, results_dir=zs_dir)

        eval_csv = evaluate_all(zs_env_for_eval, zs_t_cfg,
                                agent_cfg=zs_agent_cfg,
                                seeds=seeds, eval_episodes=eval_episodes,
                    ss_tune_env_cfg=stat_cfg,
                                results_dir=zs_dir)
        analyze_results(eval_csv, zs_dir)
        _collect_summary(zs_dir, scenario_name, "ZeroShot", nonstat_results)

    # --- Phase 3: Train specialised agents for each non-stationary ---
    print("\n>>> Phase 3: Training specialised agents <<<")
    for scenario_name, e_cfg in demand_scenarios.items():
        if scenario_name == "Stationary_Base":
            continue

        spec_dir = os.path.join(results_dir, f"{scenario_name}_Specialised")
        os.makedirs(spec_dir, exist_ok=True)

        spec_a_cfg = AgentConfig()
        spec_t_cfg = TrainConfig(episodes=episodes, results_dir=spec_dir)

        log_csv = os.path.join(spec_dir, "training_log.csv")
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["policy", "seed", "episode", "total_steps",
                             "reward", "epsilon", "loss", "duration_sec"])
            for s in seeds:
                train_seed(s, e_cfg, spec_a_cfg, spec_t_cfg, writer, f)

        eval_csv = evaluate_all(e_cfg, spec_t_cfg, agent_cfg=spec_a_cfg,
                                seeds=seeds, eval_episodes=eval_episodes,
                                results_dir=spec_dir)
        analyze_results(eval_csv, spec_dir)
        _collect_summary(spec_dir, scenario_name, "Specialised",
                         nonstat_results)

    # --- Summary ---
    out_table = os.path.join(results_dir, "nonstationary_summary.json")
    with open(out_table, "w", encoding="utf-8") as f:
        json.dump(nonstat_results, f, indent=2)

    print("\n" + "=" * 100)
    print("NON-STATIONARY DEMAND STUDY COMPLETED")
    print(f"{'Scenario':<25} | {'Training':<18} | {'Policy':<18} | "
          f"{'Mean Profit (₹)':<16} | {'Service Level':<14} | {'Avg Inv':<8}")
    print("-" * 100)
    for res in nonstat_results:
        print(f"{res['Scenario']:<25} | {res['Training']:<18} | "
              f"{res['Policy']:<18} | ₹{res['Mean_Profit']:14.2f} | "
              f"{res['Service_Level']:12.2f}% | {res['Avg_Inventory']:8.1f}")
    print("=" * 100)


def _collect_summary(
    sub_dir: str,
    scenario_name: str,
    training_label: str,
    out: List[Dict[str, Any]],
) -> None:
    """Read summary_statistics.csv and append rows to *out*."""
    summary_csv = os.path.join(sub_dir, "summary_statistics.csv")
    if not os.path.exists(summary_csv):
        return
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({
                "Scenario": scenario_name,
                "Training": training_label,
                "Policy": row["policy"],
                "Mean_Profit": float(row["profit_mean"]),
                "Service_Level": float(row["service_level_mean"]),
                "Avg_Inventory": float(row["avg_inventory_mean"]),
            })


def _append_summary_rows(
    sub_dir: str,
    scenario_name: str,
    out: List[Dict[str, Any]],
) -> None:
    """Append rows from an already-completed sensitivity run."""
    _collect_summary(sub_dir, scenario_name, "Sensitivity", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL Inventory Experiment Suite")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--nonstationary", action="store_true", help="Run non-stationary demand analysis")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes per seed")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Evaluation episodes per seed")
    args = parser.parse_args()

    if args.ablation:
        run_ablation_study(episodes=args.episodes, eval_episodes=args.eval_episodes)
    if args.sensitivity:
        run_sensitivity_study(episodes=args.episodes, eval_episodes=args.eval_episodes)
    if args.nonstationary:
        run_nonstationary_study(episodes=args.episodes, eval_episodes=args.eval_episodes)
    if not args.ablation and not args.sensitivity and not args.nonstationary:
        print("Please specify --ablation, --sensitivity, or --nonstationary.")

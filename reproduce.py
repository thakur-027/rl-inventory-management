"""One-command full pipeline reproduction script.

Executes:
1. Training across independent seeds with system profiling (env_info.json)
2. Evaluation using Common Random Numbers against all baselines (Fixed, EOQ, tuned (s,S), DP)
3. Statistical analysis: 95% CIs, paired t-tests, Wilcoxon signed-rank tests, cost breakdown
4. Publication-ready vector figure generation (.pdf and .png)
"""

from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import sys
import time
import shutil

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

from config import EnvConfig, AgentConfig, TrainConfig, BaselineConfig
from train import run_training
from evaluate import evaluate_all
from analyze import analyze_results
from plots import generate_all_plots


def run_pipeline(
    episodes: int = 500,
    eval_episodes: int = 200,
    seeds: list[int] = [42, 123, 456, 789, 1024],
    results_dir: str = "results",
    include_dp: bool = True,
    double_dqn: bool = False,
) -> None:
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir, ignore_errors=True)
        print(f"Cleaned up previous {results_dir} directory.")

    t_start = time.perf_counter()
    print("=" * 80)
    print("REINFORCEMENT LEARNING INVENTORY RESTOCKING OPTIMIZATION")
    print("End-to-End Pipeline Reproduction")
    print("=" * 80)

    e_cfg = EnvConfig()
    a_cfg = AgentConfig(double_dqn=double_dqn)
    t_cfg = TrainConfig(episodes=episodes, seeds=seeds, results_dir=results_dir)
    b_cfg = BaselineConfig()

    # 1. Train
    print("\n[STEP 1/4] Training DQN Agents across seeds...")
    run_training(e_cfg, a_cfg, t_cfg, seeds=seeds)

    # 2. Evaluate
    print("\n[STEP 2/4] Evaluating all policies with Common Random Numbers...")
    eval_csv = evaluate_all(
        env_cfg=e_cfg,
        train_cfg=t_cfg,
        agent_cfg=a_cfg,
        base_cfg=b_cfg,
        seeds=seeds,
        eval_episodes=eval_episodes,
        include_dp=include_dp,
        results_dir=results_dir,
    )

    # 3. Analyze
    print("\n[STEP 3/4] Performing statistical significance analysis...")
    analyze_results(eval_csv, results_dir=results_dir)

    # 4. Plots
    print("\n[STEP 4/4] Generating publication-grade vector figures (.pdf & .png)...")
    generate_all_plots(results_dir=results_dir)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION COMPLETED IN {elapsed:.2f} SECONDS ({elapsed/60:.2f} MINUTES)!")
    print(f"All artifacts, tables, logs, and figures saved to: {results_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce RL Inventory Management Results")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run (2 seeds x 100 episodes)")
    parser.add_argument("--episodes", type=int, default=None, help="Custom training episodes")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Custom evaluation episodes")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Custom seeds")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--include-dp", action="store_true", default=False, help="Include DP Value Iteration baseline")
    parser.add_argument("--double-dqn", action="store_true", help="Train Double DQN variant")
    args = parser.parse_args()

    if args.quick:
        episodes = 100
        eval_episodes = 50
        seeds = [42, 123]
    else:
        episodes = args.episodes or 500
        eval_episodes = args.eval_episodes or 200
        seeds = args.seeds or [42, 123, 456, 789, 1024]

    run_pipeline(
        episodes=episodes,
        eval_episodes=eval_episodes,
        seeds=seeds,
        results_dir=args.results_dir,
        include_dp=args.include_dp,
        double_dqn=args.double_dqn,
    )

"""One-command full pipeline reproduction script.

Executes:
1. Training across independent seeds with system profiling (env_info.json)
2. Evaluation using Common Random Numbers against all baselines (Fixed, EOQ, tuned (s,S))
3. Statistical analysis: 95% CIs, paired t-tests, Wilcoxon signed-rank tests, cost breakdown
4. Publication-ready vector figure generation (.pdf and .png)
"""

from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import contextlib
import sys
import time

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


class Tee:
    """Write console output to both the terminal and a run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_pipeline(
    episodes: int = 500,
    eval_episodes: int = 200,
    seeds: list[int] = [42, 123, 456, 789, 1024],
    results_dir: str = "results",
    double_dqn: bool = False,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    for filename in (
        "env_info.json", "training_log.csv", "evaluation_results.csv",
        "summary_statistics.csv", "hypothesis_tests.json", "tuned_ss.json",
    ):
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            os.remove(path)
    print(f"Reusing experiment directories under {results_dir}; refreshed main-run files only.")

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

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, log_file)), contextlib.redirect_stderr(Tee(sys.__stderr__, log_file)):
            run_pipeline(
                episodes=episodes,
                eval_episodes=eval_episodes,
                seeds=seeds,
                results_dir=args.results_dir,
                double_dqn=args.double_dqn,
            )

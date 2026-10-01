"""Training script for DQN Inventory Restocking Agent.

Features:
- Full reproducibility: seeds Python, NumPy, environment, and networks
- System environment metadata logging to `env_info.json`
- Supports multi-seed runs (default: 5 seeds × 500 episodes)
- Per-step experience replay and target network synchronization
- Outputs per-episode training log CSV (seed, episode, reward, epsilon, loss, duration)
- Saves model weights per seed to `results/`
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from typing import Dict, Any, List, Optional

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

# Force single-threaded CPU consistency
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from config import EnvConfig, AgentConfig, TrainConfig
from env import InventoryEnv
from agent import DQNAgent


# =====================================================================
# System Environment Profiler
# =====================================================================

def record_env_info(output_path: str) -> Dict[str, Any]:
    """Capture and record hardware, OS, and software environment metadata."""
    info: Dict[str, Any] = {
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "gpu_enabled": False,
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "default"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "default"),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except ImportError:
        info["scipy_version"] = "Not Installed"
        
    try:
        import matplotlib
        info["matplotlib_version"] = matplotlib.__version__
    except ImportError:
        info["matplotlib_version"] = "Not Installed"
        
    try:
        if platform.system() == "Windows":
            import subprocess
            cpu_name = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split("\n")[1].strip()
            info["processor"] = cpu_name
    except Exception:
        pass

    # Attempt to query RAM on Windows or Linux
    try:
        if platform.system() == "Windows":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["total_ram_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 2)
        else:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            info["total_ram_gb"] = round(total_bytes / (1024 ** 3), 2)
    except Exception:
        info["total_ram_gb"] = "Unknown"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return info


# =====================================================================
# Training Loop
# =====================================================================

def train_seed(
    seed: int,
    env_cfg: EnvConfig,
    agent_cfg: AgentConfig,
    train_cfg: TrainConfig,
    csv_writer: csv.writer,
    log_file,
) -> float:
    """Train DQN agent on a single seed across specified episodes."""
    random.seed(seed)
    np.random.seed(seed)

    env = InventoryEnv(env_cfg)
    agent = DQNAgent(env_cfg=env_cfg, agent_cfg=agent_cfg, seed=seed)

    start_time = time.perf_counter()
    prefix = "DoubleDQN" if agent_cfg.double_dqn else "DQN"
    print(f"\n--- Starting {prefix} Training [Seed {seed}] for {train_cfg.episodes} episodes ---")

    for ep in range(1, train_cfg.episodes + 1):
        ep_seed = seed * 100000 + ep
        obs, _ = env.reset(seed=ep_seed)

        ep_reward = 0.0
        losses: List[float] = []
        ep_start_t = time.perf_counter()

        done = False
        while not done:
            action = agent.act(obs, explore=True)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            loss = agent.step_learn(obs, action, reward, next_obs, term)
            if loss is not None:
                losses.append(loss)

            ep_reward += reward
            obs = next_obs

        ep_duration = time.perf_counter() - ep_start_t
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Log episode metrics
        csv_writer.writerow([
            prefix,
            seed,
            ep,
            agent.total_steps,
            f"{ep_reward:.2f}",
            f"{agent.epsilon:.4f}",
            f"{avg_loss:.6f}",
            f"{ep_duration:.4f}",
        ])
        log_file.flush()

        if ep % train_cfg.log_interval == 0 or ep == train_cfg.episodes:
            elapsed = time.perf_counter() - start_time
            print(
                f"[{prefix} Seed {seed}] Ep {ep:3d}/{train_cfg.episodes:3d} | "
                f"Reward: INR {ep_reward:9.2f} | "
                f"eps: {agent.epsilon:6.3f} | "
                f"Loss: {avg_loss:8.4f} | "
                f"Elapsed: {elapsed:5.1f}s"
            )

    total_training_time = time.perf_counter() - start_time

    # Save weights if requested
    if train_cfg.save_weights:
        model_name = f"{prefix.lower()}_seed_{seed}.npz"
        weights_path = os.path.join(train_cfg.results_dir, model_name)
        agent.save(weights_path)
        print(f"Saved model weights to {weights_path}")

    return total_training_time


def run_training(
    env_cfg: Optional[EnvConfig] = None,
    agent_cfg: Optional[AgentConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    seeds: Optional[List[int]] = None,
) -> None:
    env_cfg = env_cfg or EnvConfig()
    agent_cfg = agent_cfg or AgentConfig()
    train_cfg = train_cfg or TrainConfig()

    active_seeds = seeds if seeds is not None else train_cfg.seeds
    os.makedirs(train_cfg.results_dir, exist_ok=True)

    # 1. Record environment info
    env_info_path = os.path.join(train_cfg.results_dir, "env_info.json")
    env_info = record_env_info(env_info_path)
    print(f"Recorded system environment info to {env_info_path}")
    print(f"OS: {env_info['os']} | CPU: {env_info['processor']} ({env_info['cpu_count']} cores) | RAM: {env_info.get('total_ram_gb')} GB")

    # 2. Open training log CSV
    log_csv_path = os.path.join(train_cfg.results_dir, "training_log.csv")

    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "seed", "episode", "total_steps", "reward", "epsilon", "loss", "duration_sec"
        ])

        seed_times = []
        for s in active_seeds:
            t_sec = train_seed(s, env_cfg, agent_cfg, train_cfg, writer, f)
            seed_times.append(t_sec)

    print(f"\nAll training runs complete!")
    print(f"Mean training time per seed: {np.mean(seed_times):.2f}s (± {np.std(seed_times):.2f}s)")
    print(f"Training logs saved to: {log_csv_path}")

    env_info["mean_training_time_sec"] = np.mean(seed_times)
    with open(env_info_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)


# =====================================================================
# CLI Entrypoint
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent for Inventory Optimization")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to train")
    parser.add_argument("--double-dqn", action="store_true", help="Enable Double DQN")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for output")
    args = parser.parse_args()

    e_cfg = EnvConfig()
    a_cfg = AgentConfig(double_dqn=args.double_dqn, learning_rate=args.lr)
    t_cfg = TrainConfig(episodes=args.episodes, results_dir=args.results_dir)

    run_training(e_cfg, a_cfg, t_cfg, seeds=args.seeds)

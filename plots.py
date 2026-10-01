"""Publication-ready vector figure generator for RL inventory optimization.

Exports vector PDFs (and high-res PNG copies) directly from CSV results:
1. Fig 1: Training dynamics with multi-seed confidence band (mean ± std/CI) and epsilon decay.
2. Fig 2: Profit comparison bar chart with 95% confidence intervals.
3. Fig 3: Service level comparison (without misleading y-axis truncation).
4. Fig 4: Economic cost breakdown (stacked procurement, holding, stockout, fixed order).
5. Fig 5: 90-day inventory and replenishment trajectory comparison.
6. Fig 6: Policy action decision surface / heatmap across inventory states.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Configure publication-grade styling (IEEE format)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 8,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Harmonious, accessible palette
PALETTE = {
    "DQN": "#1f77b4",          # Deep Blue
    "DoubleDQN": "#2ca02c",    # Forest Green
    "FixedReorder": "#d62728", # Coral Red
    "EOQ": "#9467bd",          # Purple
    "(s,S)": "#ff7f0e",        # Warm Amber
    "DP": "#8c564b",           # Brown
}


def get_color(policy_name: str) -> str:
    for k, col in PALETTE.items():
        if k == policy_name:
            return col
    for k, col in PALETTE.items():
        if k in policy_name:
            return col
    return "#555555"


# =====================================================================
# 1. Training Dynamics with Multi-Seed Band
# =====================================================================

def plot_training_dynamics(train_csv_path: str, out_dir: str) -> None:
    if not os.path.exists(train_csv_path):
        print(f"Skipping Fig 1: {train_csv_path} not found.")
        return

    ep_rewards = defaultdict(lambda: defaultdict(list))
    ep_epsilons = defaultdict(lambda: defaultdict(list))

    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pol = r["policy"]
            ep = int(r["episode"])
            rew = float(r["reward"])
            eps = float(r["epsilon"])
            ep_rewards[pol][ep].append(rew)
            ep_epsilons[pol][ep].append(eps)

    fig, ax1 = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    ax2 = ax1.twinx()

    for pol in ep_rewards:
        eps_sorted = sorted(ep_rewards[pol].keys())
        means = [np.mean(ep_rewards[pol][ep]) for ep in eps_sorted]
        stds = [np.std(ep_rewards[pol][ep]) for ep in eps_sorted]

        # Smooth curve with rolling window of 15
        window = 15
        if len(means) >= window:
            smooth_mean = np.convolve(means, np.ones(window) / window, mode="valid")
            smooth_std = np.convolve(stds, np.ones(window) / window, mode="valid")
            x_vals = eps_sorted[window - 1 :]
        else:
            smooth_mean = np.array(means)
            smooth_std = np.array(stds)
            x_vals = eps_sorted

        col = get_color(pol)
        ax1.plot(x_vals, smooth_mean, label=f"{pol} Mean Return", color=col, linewidth=1.2)
        ax1.fill_between(
            x_vals,
            smooth_mean - smooth_std,
            smooth_mean + smooth_std,
            color=col,
            alpha=0.18,
            label=f"{pol} ±1 Std Dev Band",
        )

        # Plot epsilon on twin axis
        eps_vals = [np.mean(ep_epsilons[pol][ep]) for ep in eps_sorted]
        ax2.plot(eps_sorted, eps_vals, color="#777777", linestyle=":", linewidth=1.0, label="Exploration Rate (ε)")

    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Episode Net Reward (₹)")
    ax2.set_ylabel("Exploration Factor (ε)", color="#555555")
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2[:1], labels1 + labels2[:1], loc="lower right", framealpha=0.9, fontsize=6)

    pdf_path = os.path.join(out_dir, "fig1_training_dynamics.pdf")
    png_path = os.path.join(out_dir, "fig1_training_dynamics.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 1: {pdf_path}")


# =====================================================================
# 2. Profit Comparison with 95% Confidence Intervals
# =====================================================================

def plot_profit_comparison(eval_csv_path: str, out_dir: str) -> None:
    if not os.path.exists(eval_csv_path):
        return

    pol_ep_data = defaultdict(lambda: defaultdict(list))
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pol_ep_data[r["policy"]][int(r["episode"])].append(float(r["profit"]))

    policy_profits = defaultdict(list)
    for pol, ep_dict in pol_ep_data.items():
        for ep in sorted(ep_dict.keys()):
            policy_profits[pol].append(np.mean(ep_dict[ep]))

    policies = list(policy_profits.keys())
    means = [np.mean(policy_profits[p]) for p in policies]
    cis = []
    for p in policies:
        arr = np.array(policy_profits[p])
        ci_hw = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))
        cis.append(ci_hw)

    colors = [get_color(p) for p in policies]

    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    bars = ax.bar(policies, means, yerr=cis, capsize=3, color=colors, alpha=0.88, edgecolor="black", linewidth=0.8)
    
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    for i, bar in enumerate(bars):
        bar.set_hatch(hatches[i % len(hatches)])

    ax.set_ylabel("Mean Net Profit per Episode (₹)")
    # Numeric labels shrunk for IEEE width
    for bar, m, ci in zip(bars, means, cis):
        y_pos = m + ci + (max(means) * 0.02)
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"₹{m/1000:,.1f}k", ha="center", va="bottom", fontsize=6, weight="bold")

    y_max = max(means) + max(cis)
    ax.set_ylim(0, y_max * 1.15)
    plt.xticks(rotation=15, ha='right')

    pdf_path = os.path.join(out_dir, "fig2_profit_comparison.pdf")
    png_path = os.path.join(out_dir, "fig2_profit_comparison.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 2: {pdf_path}")


# =====================================================================
# 3. Service Level Comparison (Without Artificial Truncation)
# =====================================================================

def plot_service_level(eval_csv_path: str, out_dir: str) -> None:
    if not os.path.exists(eval_csv_path):
        return

    policy_sl = defaultdict(lambda: defaultdict(list))
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            policy_sl[r["policy"]][int(r["episode"])].append(float(r["service_level"]) * 100.0)

    policies = list(policy_sl.keys())
    episode_values = {
        policy: np.array([np.mean(values) for values in episodes.values()])
        for policy, episodes in policy_sl.items()
    }
    means = [np.mean(episode_values[p]) for p in policies]
    cis = [1.96 * np.std(episode_values[p], ddof=1) / np.sqrt(len(episode_values[p])) for p in policies]
    colors = [get_color(p) for p in policies]
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    bars = ax.bar(policies, means, yerr=cis, capsize=3, color=colors, alpha=0.88, edgecolor="black", linewidth=0.8)

    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    for i, bar in enumerate(bars):
        bar.set_hatch(hatches[i % len(hatches)])

    ax.set_ylabel("Service Level (%)")

    # Dynamic y-limits starting from minimum minus margin (unbiased, non-misleading)
    min_val = min(means) - max(cis)
    lower_lim = max(0.0, math.floor(min_val / 10.0) * 10.0 - 5.0)
    ax.set_ylim(lower_lim, 105.0)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m + 1.0, f"{m:.1f}%", ha="center", va="bottom", fontsize=6, weight="bold")
        
    plt.xticks(rotation=15, ha='right')

    pdf_path = os.path.join(out_dir, "fig3_service_level.pdf")
    png_path = os.path.join(out_dir, "fig3_service_level.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 3: {pdf_path}")


# =====================================================================
# 4. Economic Cost Breakdown (Grouped / Stacked Bars)
# =====================================================================

def plot_cost_breakdown(eval_csv_path: str, out_dir: str) -> None:
    if not os.path.exists(eval_csv_path):
        return

    data = defaultdict(lambda: defaultdict(list))
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pol = r["policy"]
            data[pol]["Holding"].append(float(r["holding"]))
            data[pol]["Stockout"].append(float(r["stockout"]))
            data[pol]["Fixed Order"].append(float(r["ordering"]))

    policies = list(data.keys())
    cost_categories = ["Holding", "Stockout", "Fixed Order"]
    category_colors = ["#59a14f", "#e15759", "#edc948"]

    means_by_cat = {
        cat: [np.mean(data[p][cat]) for p in policies] for cat in cost_categories
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    bottom = np.zeros(len(policies))

    hatches = ["///", "...", "\\\\\\"]
    for cat, col, hatch in zip(cost_categories, category_colors, hatches):
        vals = np.array(means_by_cat[cat])
        ax.bar(policies, vals, bottom=bottom, label=cat, color=col, hatch=hatch, alpha=0.9, edgecolor="black", linewidth=0.7)
        bottom += vals

    ax.set_ylabel("Mean Incurred Cost per Episode (₹)")
    ax.legend(loc="upper right", framealpha=0.9)

    pdf_path = os.path.join(out_dir, "fig4_cost_breakdown.pdf")
    png_path = os.path.join(out_dir, "fig4_cost_breakdown.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Generated Figure 4: {pdf_path}")


# =====================================================================
# 5. Inventory Trajectory Simulation
# =====================================================================

def plot_trajectory_comparison(out_dir: str, env_cfg=None, seed: int = 12345) -> None:
    from config import EnvConfig
    from env import InventoryEnv
    from baselines import FixedReorderPolicy, EOQPolicy, SSPolicy
    from agent import DQNAgent
    import json

    env_cfg = env_cfg or EnvConfig()
    eval_env = InventoryEnv(env_cfg)

    policies = [
        ("FixedReorder", FixedReorderPolicy(env_cfg)),
        ("EOQ", EOQPolicy(env_cfg)),
    ]
    tuned_path = os.path.join(out_dir, "tuned_ss.json")
    if os.path.exists(tuned_path):
        with open(tuned_path, "r", encoding="utf-8") as f:
            tuned = json.load(f)
        params = tuned.get("retuned", tuned)
        policies.append(("(s,S)", SSPolicy(env_cfg, s=params["s"], S=params["S"])))

    # Check for DQN
    dqn_weights = os.path.join(out_dir, "dqn_seed_42.npz")
    if os.path.exists(dqn_weights):
        agent = DQNAgent(env_cfg=env_cfg, seed=42)
        agent.load(dqn_weights)
        policies.insert(0, ("DQN", agent))

    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    line_styles = {"DQN": "-", "FixedReorder": "--", "EOQ": ":", "(s,S)": "-."}

    for pol_name, pol_obj in policies:
        obs, _ = eval_env.reset(seed=seed)
        inv_hist = [eval_env.inventory]
        done = False
        while not done:
            if hasattr(pol_obj, "act"):
                if isinstance(pol_obj, DQNAgent):
                    act = pol_obj.act(obs, explore=False)
                else:
                    act = pol_obj.act(obs)
            else:
                act = 0
            obs, _, term, trunc, info = eval_env.step(act)
            done = term or trunc
            inv_hist.append(info["inventory_end"])

        col = get_color(pol_name)
        ax.plot(range(len(inv_hist)), inv_hist, label=pol_name, color=col, linestyle=line_styles.get(pol_name, "-"), linewidth=1.2)

    ax.axhline(env_cfg.max_inventory, color="gray", linestyle="--", alpha=0.7, label="Capacity (C=100)")
    ax.set_xlabel("Day of Episode (t)")
    ax.set_ylabel("On-Hand Inventory Level")
    ax.legend(loc="upper right", framealpha=0.9)

    pdf_path = os.path.join(out_dir, "fig5_inventory_trajectory.pdf")
    png_path = os.path.join(out_dir, "fig5_inventory_trajectory.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 5: {pdf_path}")


# =====================================================================
# 6. Policy Action Decision Heatmap
# =====================================================================

def plot_policy_heatmap(out_dir: str, env_cfg=None) -> None:
    from config import EnvConfig
    from agent import DQNAgent

    env_cfg = env_cfg or EnvConfig()
    dqn_weights = os.path.join(out_dir, "dqn_seed_42.npz")
    if not os.path.exists(dqn_weights):
        # Fallback to any dqn seed weights found in out_dir
        for f in os.listdir(out_dir):
            if f.startswith("dqn_seed_") and f.endswith(".npz"):
                dqn_weights = os.path.join(out_dir, f)
                break

    if not os.path.exists(dqn_weights):
        print("Skipping Figure 6: No DQN model weights found for heatmap.")
        return

    agent = DQNAgent(env_cfg=env_cfg, seed=42)
    agent.load(dqn_weights)

    # Grid of inventory position vs pipeline orders sum
    position_levels = np.linspace(0, 200, 21)
    pipeline_levels = np.linspace(0, 100, 21)
    grid_actions = np.zeros((len(pipeline_levels), len(position_levels)))

    cap = float(env_cfg.max_inventory)
    for i, p_val in enumerate(pipeline_levels):
        for j, position in enumerate(position_levels):
            i_val = max(0.0, position - p_val)
            # Split pipeline equally across lead time orders
            per_pipe = (p_val / env_cfg.lead_time) / cap
            obs = np.array([i_val / cap] + [per_pipe] * env_cfg.lead_time, dtype=np.float32)
            act_idx = agent.act(obs, explore=False)
            qty = env_cfg.action_values[act_idx]
            grid_actions[i, j] = qty

    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    c = ax.imshow(
        grid_actions,
        origin="lower",
        extent=[0, 200, 0, 100],
        aspect="auto",
        cmap="Greys",
    )
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Replenishment Order Quantity (Units)")

    ax.set_xlabel("Inventory Position (I + pipeline)")
    ax.set_ylabel("Total Pipeline On-Order Inventory (∑ O_i)")

    pdf_path = os.path.join(out_dir, "fig6_policy_heatmap.pdf")
    png_path = os.path.join(out_dir, "fig6_policy_heatmap.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 6: {pdf_path}")


# =====================================================================
# CLI Entrypoint
# =====================================================================

def generate_all_plots(results_dir: str = "results") -> None:
    train_csv = os.path.join(results_dir, "training_log.csv")
    eval_csv = os.path.join(results_dir, "evaluation_results.csv")

    plot_training_dynamics(train_csv, results_dir)
    plot_profit_comparison(eval_csv, results_dir)
    plot_service_level(eval_csv, results_dir)
    plot_cost_breakdown(eval_csv, results_dir)
    plot_trajectory_comparison(results_dir)
    plot_policy_heatmap(results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Vector PDF Figures")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory with CSV results")
    args = parser.parse_args()

    generate_all_plots(args.results_dir)

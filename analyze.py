"""Statistical analysis pipeline for evaluation results.

Calculates:
- Mean, standard deviation, median, IQR, and 95% confidence intervals
- Comprehensive breakdown: Profit, Revenue, Procurement, Holding, Stockout,
  Ordering costs, Order count, Average inventory, Unmet demand, Service level
- Paired statistical significance tests (Paired t-test, Wilcoxon signed-rank)
  enabled by Common Random Numbers evaluation design
- Cohen's d effect sizes and empirical win-rates
- Per-seed variability analysis
- Exports summary CSV and hypothesis test JSON reports
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

# Try importing scipy; provide robust pure-numpy fallbacks if unavailable
try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =====================================================================
# Statistical Helpers
# =====================================================================

def compute_ci95(data: np.ndarray) -> Tuple[float, float, float]:
    """Compute mean and 95% confidence interval half-width."""
    n = len(data)
    if n <= 1:
        return float(np.mean(data)), 0.0, 0.0
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    se = std / math.sqrt(n)

    if HAS_SCIPY:
        t_crit = float(sp_stats.t.ppf(0.975, df=n - 1))
    else:
        # Standard normal approx for large n (n >= 30)
        t_crit = 1.96

    half_width = t_crit * se
    return mean, mean - half_width, mean + half_width


def paired_t_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Paired Student's t-test returning (t_stat, p_val)."""
    diff = x - y
    n = len(diff)
    if n <= 1:
        return 0.0, 1.0

    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    if std_d == 0.0:
        return 0.0, 1.0

    t_stat = mean_d / (std_d / math.sqrt(n))

    if HAS_SCIPY:
        res = sp_stats.ttest_rel(x, y)
        return float(res.statistic), float(res.pvalue)

    # Pure Python / NumPy two-tailed normal approximation for large n
    # 2 * (1 - Phi(|t|))
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
    return float(t_stat), float(p_val)


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test returning (stat, p_val)."""
    if HAS_SCIPY:
        try:
            res = sp_stats.wilcoxon(x, y, zero_method="pratt")
            return float(res.statistic), float(res.pvalue)
        except Exception:
            pass

    # Pure NumPy implementation of Wilcoxon signed-rank test
    diff = x - y
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0, 1.0

    abs_diff = np.abs(diff)
    order = np.argsort(abs_diff)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + 1)

    # Handle ties in ranks
    unique_d, counts = np.unique(abs_diff, return_counts=True)
    for val, count in zip(unique_d, counts):
        if count > 1:
            indices = np.where(abs_diff == val)[0]
            ranks[indices] = float(np.mean(ranks[indices]))

    w_pos = float(np.sum(ranks[diff > 0]))
    w_neg = float(np.sum(ranks[diff < 0]))
    w_stat = min(w_pos, w_neg)

    # Normal approximation for n >= 20
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_stat - mean_w) / math.sqrt(var_w)
    p_val = 2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return float(w_stat), float(min(1.0, max(0.0, p_val)))


def tost_paired(x: np.ndarray, y: np.ndarray, margin: float) -> float:
    """Two One-Sided Tests (TOST) for equivalence."""
    # Test 1: H0: mean(x) - mean(y) <= -margin -> Ha: mean(x) - mean(y) > -margin
    t1, p_two_sided1 = paired_t_test(x, y - margin)
    p1 = p_two_sided1 / 2.0 if t1 > 0 else 1.0 - (p_two_sided1 / 2.0)
    
    # Test 2: H0: mean(x) - mean(y) >= margin -> Ha: mean(x) - mean(y) < margin
    t2, p_two_sided2 = paired_t_test(x, y + margin)
    p2 = p_two_sided2 / 2.0 if t2 < 0 else 1.0 - (p_two_sided2 / 2.0)
    
    return float(max(p1, p2))

def holm_bonferroni(p_values: List[float]) -> List[float]:
    n = len(p_values)
    sorted_indices = sorted(range(n), key=lambda i: p_values[i])
    adj_p = [0.0] * n
    current_max = 0.0
    for i, idx in enumerate(sorted_indices):
        adj = p_values[idx] * (n - i)
        current_max = max(current_max, adj)
        adj_p[idx] = min(1.0, current_max)
    return adj_p

# =====================================================================
# Main Analysis Pipeline
# =====================================================================

def analyze_results(csv_path: str, results_dir: str = "results") -> None:
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return
    os.makedirs(results_dir, exist_ok=True)

    # Read rows
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append({
                "policy": r["policy"],
                "seed": int(r["seed"]) if r["seed"] != "Baseline" else "Baseline",
                "episode": int(r["episode"]),
                "profit": float(r["profit"]),
                "revenue": float(r["revenue"]),
                "procurement": float(r["procurement"]),
                "holding": float(r["holding"]),
                "stockout": float(r["stockout"]),
                "ordering": float(r["ordering"]),
                "number_of_orders": int(r["number_of_orders"]),
                "average_inventory": float(r["average_inventory"]),
                "unmet": int(r["unmet"]),
                "demand": int(r["demand"]),
                "discarded_stock": int(float(r.get("discarded_stock", 0))),
                "service_level": float(r["service_level"]) * 100.0,  # in %
                "decision_time_us": float(r.get("decision_time_us", 0.0)),
            })

    if not records:
        print("No evaluation records found.")
        return

    metric_keys = [
        "profit", "revenue", "procurement", "holding", "stockout",
        "ordering", "number_of_orders", "average_inventory", "unmet", "demand",
        "discarded_stock", "service_level", "decision_time_us"
    ]

    # Group by policy and episode, averaging over seeds
    pol_ep_data = defaultdict(lambda: defaultdict(list))
    for r in records:
        pol_ep_data[r["policy"]][r["episode"]].append(r)

    policy_data = defaultdict(lambda: defaultdict(list))
    for pol, ep_dict in pol_ep_data.items():
        for ep in sorted(ep_dict.keys()):
            rows = ep_dict[ep]
            for k in metric_keys:
                avg_val = np.mean([r[k] for r in rows])
                policy_data[pol][k].append(avg_val)

    # 1. Summary Statistics Table
    summary_rows = []
    print("\n" + "=" * 105)
    print("TABLE I: OVERALL PERFORMANCE SUMMARY (Seed-averaged, n = 200 paired episodes)")
    print("=" * 105)
    header = (
        f"{'Policy':<22} | {'Profit (INR) [95% CI]':<26} | {'Std Dev':<9} | "
        f"{'Service Lvl':<11} | {'Avg Inv':<8} | {'Orders':<6} | {'Latency':<8}"
    )
    print(header)
    print("-" * 105)

    metric_keys_no_time = [k for k in metric_keys if k != "decision_time_us"]

    for pol, metrics in policy_data.items():
        p_arr = np.array(metrics["profit"])
        p_mean, p_lo, p_hi = compute_ci95(p_arr)
        p_std = float(np.std(p_arr, ddof=1))
        p_med = float(np.median(p_arr))
        p_iqr = float(np.percentile(p_arr, 75) - np.percentile(p_arr, 25))

        sl_arr = np.array(metrics["service_level"])
        sl_mean = float(np.mean(sl_arr))

        inv_arr = np.array(metrics["average_inventory"])
        inv_mean = float(np.mean(inv_arr))

        ord_arr = np.array(metrics["number_of_orders"])
        ord_mean = float(np.mean(ord_arr))

        lat_arr = np.array(metrics["decision_time_us"])
        lat_mean = float(np.mean(lat_arr))

        profit_str = f"INR {p_mean:,.0f} [{p_lo:,.0f}, {p_hi:,.0f}]"
        print(
            f"{pol:<22} | {profit_str:<26} | {p_std:8.1f} | "
            f"{sl_mean:9.2f}% | {inv_mean:7.1f} | {ord_mean:5.1f} | {lat_mean:6.1f}us"
        )

        row_dict = {
            "policy": pol,
            "profit_mean": p_mean,
            "profit_ci_lower": p_lo,
            "profit_ci_upper": p_hi,
            "profit_std": p_std,
            "profit_median": p_med,
            "profit_iqr": p_iqr,
            "service_level_mean": sl_mean,
            "avg_inventory_mean": inv_mean,
            "orders_count_mean": ord_mean,
            "latency_us_mean": lat_mean,
        }
        for k in metric_keys_no_time:
            row_dict[f"{k}_mean"] = float(np.mean(metrics[k]))
            row_dict[f"{k}_std"] = float(np.std(metrics[k], ddof=1) if len(metrics[k]) > 1 else 0.0)

        summary_rows.append(row_dict)

    print("=" * 105)

    # 2. Detailed Cost Breakdown Table
    print("\n" + "=" * 105)
    print("TABLE II: DETAILED ECONOMIC COST BREAKDOWN (Mean INR per 90-day Episode)")
    print("=" * 105)
    cost_header = (
        f"{'Policy':<22} | {'Revenue':<11} | {'Procurement':<11} | "
        f"{'Holding':<10} | {'Stockout':<10} | {'Fixed Order':<11} | {'Net Profit':<11}"
    )
    print(cost_header)
    print("-" * 105)
    for pol, metrics in policy_data.items():
        rev = np.mean(metrics["revenue"])
        proc = np.mean(metrics["procurement"])
        hold = np.mean(metrics["holding"])
        stk = np.mean(metrics["stockout"])
        fix = np.mean(metrics["ordering"])
        prof = np.mean(metrics["profit"])
        print(
            f"{pol:<22} | {rev:10.1f} | {proc:10.1f} | "
            f"{hold:9.1f} | {stk:9.1f} | {fix:10.1f} | {prof:10.1f}"
        )
    print("=" * 105)

    # 3. Paired Hypothesis Tests vs the explicitly selected learned policy.
    if "DQN" in policy_data:
        primary_policy = "DQN"
    elif "DoubleDQN" in policy_data:
        primary_policy = "DoubleDQN"
    else:
        raise ValueError("Evaluation results must contain DQN or DoubleDQN records.")
    hypothesis_results: Dict[str, Any] = {}

    if primary_policy in policy_data:
        dqn_profits = np.array(policy_data[primary_policy]["profit"])

        print(f"\n" + "=" * 105)
        print(f"TABLE III: PAIRED STATISTICAL TESTS vs {primary_policy} (Common Random Numbers)")
        print("=" * 105)
        test_header = (
            f"{'Baseline Policy':<24} | {'Mean Diff (INR)':<15} | {'Paired t-stat':<13} | "
            f"{'p-value (t)':<12} | {'Wilcoxon p':<11} | {'Cohen d_z':<9} | {'Win Rate':<8}"
        )
        print(test_header)
        print("-" * 105)

        test_results = []
        for base_pol, metrics in policy_data.items():
            if base_pol == primary_policy:
                continue

            base_profits = np.array(metrics["profit"])
            min_len = min(len(dqn_profits), len(base_profits))
            d_x = dqn_profits[:min_len]
            d_y = base_profits[:min_len]

            diff = d_x - d_y
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))
            cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0

            t_stat, p_val_t = paired_t_test(d_x, d_y)
            w_stat, p_val_w = wilcoxon_test(d_x, d_y)
            win_rate = float(np.mean(d_x > d_y) * 100.0)
            margin = 0.01 * float(np.mean(d_y))
            p_tost = tost_paired(d_x, d_y, margin)

            test_results.append({
                "base_pol": base_pol,
                "mean_diff": mean_diff,
                "t_stat": t_stat,
                "p_val_t": p_val_t,
                "w_stat": w_stat,
                "p_val_w": p_val_w,
                "cohen_d": cohen_d,
                "win_rate": win_rate,
                "p_tost": p_tost,
            })

        raw_p_t = [res["p_val_t"] for res in test_results]
        raw_p_w = [res["p_val_w"] for res in test_results]
        adj_p_t = holm_bonferroni(raw_p_t)
        adj_p_w = holm_bonferroni(raw_p_w)

        def fmt_p(p):
            return "< 0.001" if p < 0.001 else f"{p:.3f}"

        for i, res in enumerate(test_results):
            print(
                f"{res['base_pol']:<24} | {res['mean_diff']:+13.1f} | {res['t_stat']:12.3f} | "
                f"{fmt_p(adj_p_t[i]):>12} | {fmt_p(adj_p_w[i]):>11} | {res['cohen_d']:8.2f} | {res['win_rate']:6.1f}%"
            )

            hypothesis_results[res["base_pol"]] = {
                "mean_profit_difference": res["mean_diff"],
                "primary_policy_profit_improvement_percent": (
                    res["mean_diff"] / abs(float(np.mean(base_profits))) * 100.0
                    if np.mean(base_profits) != 0 else 0.0
                ),
                "t_statistic": res["t_stat"],
                "p_value_paired_t_adj": adj_p_t[i],
                "wilcoxon_statistic": res["w_stat"],
                "p_value_wilcoxon_adj": adj_p_w[i],
                "cohen_d": res["cohen_d"],
                "dqn_win_rate_percent": res["win_rate"],
                "statistically_significant_05": adj_p_t[i] < 0.05,
                "p_value_tost_1pct": res["p_tost"],
            }
            if res["base_pol"].startswith("(s,S)"):
                print(f"\n[TOST Equivalence] {primary_policy} vs {res['base_pol']} (±1% margin): p-value = {res['p_tost']:.4f}")
        print("=" * 105)

    # Add percentage improvements against every baseline to the summary.
    baseline_means = {
        row["policy"]: row["profit_mean"]
        for row in summary_rows
        if row["policy"] not in ("DQN", "DoubleDQN")
    }
    for row in summary_rows:
        for baseline, baseline_mean in baseline_means.items():
            if baseline_mean:
                safe_name = "".join(ch if ch.isalnum() else "_" for ch in baseline).strip("_").lower()
                row[f"profit_improvement_vs_{safe_name}_percent"] = (
                    (row["profit_mean"] - baseline_mean) / abs(baseline_mean) * 100.0
                )

    # 4. Per-Seed Breakdown
    print(f"\n" + "=" * 105)
    print("TABLE IV: PER-SEED PROFIT BREAKDOWN (INR Mean +/- Std across Seeds)")
    print("=" * 105)

    # Group by seed and policy
    seed_pol_profit = defaultdict(lambda: defaultdict(list))
    all_seeds = sorted(list({r["seed"] for r in records if r["seed"] != "Baseline"}))
    all_pols = [p for p in policy_data.keys() if any(r["policy"] == p and r["seed"] != "Baseline" for r in records)]

    for r in records:
        if r["seed"] != "Baseline":
            seed_pol_profit[r["seed"]][r["policy"]].append(r["profit"])

    seed_header = f"{'Seed':<8} | " + " | ".join(f"{p:<18}" for p in all_pols)
    print(seed_header)
    print("-" * 105)
    for s in all_seeds:
        row_str = f"{s:<8} | "
        vals = []
        for p in all_pols:
            arr = seed_pol_profit[s].get(p, [])
            val = f"INR {np.mean(arr):,.0f}" if arr else "N/A"
            vals.append(f"{val:<18}")
        print(row_str + " | ".join(vals))
    print("=" * 105)

    per_seed_rows = []
    for policy in all_pols:
        seed_means = [
            float(np.mean(seed_pol_profit[s][policy]))
            for s in all_seeds
            if seed_pol_profit[s].get(policy)
        ]
        per_seed_rows.append({
            "policy": policy,
            "profit_mean_across_seeds": float(np.mean(seed_means)) if seed_means else 0.0,
            "profit_std_across_seeds": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
            "n_seeds": len(seed_means),
        })
    per_seed_path = os.path.join(results_dir, "per_seed_profit.csv")
    with open(per_seed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    print(f"Saved per-seed profit table to: {per_seed_path}")

    # Save summary CSV
    summary_csv_path = os.path.join(results_dir, "summary_statistics.csv")
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSaved summary statistics to: {summary_csv_path}")

    # Save hypothesis JSON
    hypothesis_json_path = os.path.join(results_dir, "hypothesis_tests.json")
    with open(hypothesis_json_path, "w", encoding="utf-8") as f:
        json.dump(hypothesis_results, f, indent=2)
    print(f"Saved hypothesis testing report to: {hypothesis_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Inventory Evaluation CSV Results")
    parser.add_argument("--csv", type=str, default="results/evaluation_results.csv", help="Input CSV path")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save summaries")
    args = parser.parse_args()

    analyze_results(args.csv, args.results_dir)

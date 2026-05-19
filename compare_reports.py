#!/usr/bin/env python3
"""
Compare program durations across scheduling strategies from inference-perf reports.

Produces a CDF of session duration with lines grouped by LLM call count quartiles
and differentiated by strategy via line style.

Reads per_session_lifecycle_metrics.json from each report directory.

Usage:
    python3 compare_reports.py \
        --drr reports-drr/reports \
        --las reports-las/reports \
        --rr  reports-rr/reports \
        -o comparison_plots/
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRATEGY_STYLES = {
    "DRR": {"linestyle": "-",  "marker": "o", "markevery": 30},
    "LAS": {"linestyle": "--", "marker": "s", "markevery": 30},
    "RR":  {"linestyle": ":",  "marker": "D", "markevery": 30},
}

GROUP_COLORS = plt.cm.tab10.colors


def load_per_session_metrics(report_dir: str) -> List[dict]:
    """Load per_session_lifecycle_metrics.json from a report directory."""
    path = os.path.join(report_dir, "per_session_lifecycle_metrics.json")
    if not os.path.exists(path):
        print(f"[compare_reports] ERROR: {path} not found. "
              f"Ensure per_session: true is set in the report config.")
        return []
    with open(path) as f:
        return json.load(f)


def load_strategy_metrics(report_dir: str) -> Dict[int, dict]:
    """Return {session_index: {"duration": ..., "requests": ...}} from per-session data."""
    sessions = load_per_session_metrics(report_dir)
    metrics = {}
    for i, s in enumerate(sessions):
        metrics[i] = {
            "duration": s["duration_sec"],
            "requests": s["num_events"],
        }
    return metrics


def build_request_groups(req_totals: Dict[int, int]) -> Tuple[Dict[int, str], Dict[str, tuple]]:
    """Quartile-based grouping by num_events (LLM call count)."""
    vals = sorted(req_totals.values())
    n = len(vals)
    if n == 0:
        return {}, {}

    q1, q2, q3 = vals[n // 4], vals[n // 2], vals[3 * n // 4]
    unique_edges = sorted(set([vals[0], q1, q2, q3, vals[-1]]))

    if len(unique_edges) < 3:
        label = f"{int(vals[0])}-{int(vals[-1])} reqs"
        return {pid: label for pid in req_totals}, {label: GROUP_COLORS[0]}

    bins = []
    for i in range(len(unique_edges) - 1):
        lo, hi = unique_edges[i], unique_edges[i + 1]
        bins.append((lo, hi, f"{int(lo)}-{int(hi)} reqs", i == len(unique_edges) - 2))

    group_to_color = {label: GROUP_COLORS[i % len(GROUP_COLORS)]
                      for i, (_, _, label, _) in enumerate(bins)}

    pid_to_group = {}
    for pid, total in req_totals.items():
        for lo, hi, label, is_last in bins:
            if (is_last and total >= lo) or (not is_last and lo <= total < hi):
                pid_to_group[pid] = label
                break
        else:
            pid_to_group[pid] = bins[-1][2]

    return pid_to_group, group_to_color


def plot_duration_cdf(strategy_metrics: Dict[str, Dict[int, dict]],
                      pid_to_group: Dict[int, str],
                      group_to_color: Dict[str, tuple],
                      out_path: str):
    """CDF of session duration: one line per (strategy, group) pair."""
    fig, ax = plt.subplots(figsize=(14, 7))
    groups_sorted = sorted(group_to_color.keys(),
                           key=lambda g: int(g.split("-")[0]))

    for strategy, metrics in sorted(strategy_metrics.items()):
        style = STRATEGY_STYLES.get(strategy, {"linestyle": "-", "marker": "x", "markevery": 30})
        for group in groups_sorted:
            stage_ids = [sid for sid in metrics if pid_to_group.get(sid) == group]
            if not stage_ids:
                continue
            vals = sorted(metrics[sid]["duration"] for sid in stage_ids)
            ys = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, ys,
                    color=group_to_color[group],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markevery=style["markevery"],
                    markersize=5,
                    linewidth=1.8,
                    label=f"{strategy} — {group} ({len(vals)})",
                    alpha=0.85)

    ax.set_xlabel("Session Duration (s)", fontsize=11)
    ax.set_ylabel("CDF (fraction of sessions)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("CDF of Session Duration by Strategy & Request-Count Group", fontsize=12)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1,
              title="Strategy — Group (n)", title_fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_reports] Wrote {out_path}")


def plot_duration_cdf_per_category(strategy_metrics: Dict[str, Dict[int, dict]],
                                   pid_to_group: Dict[int, str],
                                   group_to_color: Dict[str, tuple],
                                   out_dir: str):
    """One separate plot per session category so small-session differences are visible."""
    groups_sorted = sorted(group_to_color.keys(),
                           key=lambda g: int(g.split("-")[0]))

    strategy_colors = {"DRR": plt.cm.Set2.colors[0],
                       "LAS": plt.cm.Set2.colors[1],
                       "RR":  plt.cm.Set2.colors[2]}

    for group in groups_sorted:
        fig, ax = plt.subplots(figsize=(8, 5))
        n_in_group = sum(1 for s in pid_to_group.values() if s == group)

        for strategy, metrics in sorted(strategy_metrics.items()):
            style = STRATEGY_STYLES.get(strategy, {"linestyle": "-", "marker": "x", "markevery": 10})
            stage_ids = [sid for sid in metrics if pid_to_group.get(sid) == group]
            if not stage_ids:
                continue
            vals = sorted(metrics[sid]["duration"] for sid in stage_ids)
            ys = np.arange(1, len(vals) + 1) / len(vals)
            me = max(1, len(vals) // 10)
            ax.plot(vals, ys,
                    color=strategy_colors.get(strategy, "gray"),
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markevery=me,
                    markersize=5,
                    linewidth=1.8,
                    label=strategy,
                    alpha=0.85)

        ax.set_title(f"CDF of Session Duration — {group} (n={n_in_group})", fontsize=11)
        ax.set_xlabel("Session Duration (s)", fontsize=10)
        ax.set_ylabel("CDF", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        safe_name = group.replace(" ", "_")
        path = os.path.join(out_dir, f"program_duration_cdf_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[compare_reports] Wrote {path}")


def plot_avg_request_completion(strategy_metrics: Dict[str, Dict[int, dict]],
                                out_path: str):
    """Bar chart of average per-request completion time for each strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = sorted(strategy_metrics.keys())
    colors = plt.cm.Set2.colors

    avgs = []
    for name in names:
        per_req = [m["duration"] / m["requests"]
                   for m in strategy_metrics[name].values() if m["requests"] > 0]
        avgs.append(sum(per_req) / len(per_req) if per_req else 0)

    bars = ax.bar(names, avgs, color=[colors[i % len(colors)] for i in range(len(names))],
                  edgecolor="white", width=0.45)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}s", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Avg Per-Request Completion Time (s)", fontsize=11)
    ax.set_title("Average Request Completion Time by Strategy", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_reports] Wrote {out_path}")


def plot_sum_program_duration(strategy_metrics: Dict[str, Dict[int, dict]],
                              out_path: str):
    """Bar chart of sum of all session durations for each strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = sorted(strategy_metrics.keys())
    colors = plt.cm.Set2.colors

    sums = []
    for name in names:
        sums.append(sum(m["duration"] for m in strategy_metrics[name].values()))

    bars = ax.bar(names, sums, color=[colors[i % len(colors)] for i in range(len(names))],
                  edgecolor="white", width=0.45)
    for bar, val in zip(bars, sums):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,.0f}s", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Sum of Session Durations (s)", fontsize=11)
    ax.set_title("Total Session Duration by Strategy", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_reports] Wrote {out_path}")


def plot_avg_program_duration(strategy_metrics: Dict[str, Dict[int, dict]],
                              out_path: str):
    """Bar chart of average session duration for each strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = sorted(strategy_metrics.keys())
    colors = plt.cm.Set2.colors

    avgs = []
    for name in names:
        durs = [m["duration"] for m in strategy_metrics[name].values()]
        avgs.append(sum(durs) / len(durs) if durs else 0)

    bars = ax.bar(names, avgs, color=[colors[i % len(colors)] for i in range(len(names))],
                  edgecolor="white", width=0.45)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}s", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Avg Session Duration (s)", fontsize=11)
    ax.set_title("Average Session Duration by Strategy", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_reports] Wrote {out_path}")


def print_percentile_table(strategy_metrics: Dict[str, Dict[int, dict]],
                           pid_to_group: Dict[int, str],
                           group_to_color: Dict[str, tuple],
                           out_path: str = None):
    """Print and optionally save p50/p99 of session duration per category per strategy."""
    groups_sorted = sorted(group_to_color.keys(),
                           key=lambda g: int(g.split("-")[0]))
    strategies = sorted(strategy_metrics.keys())

    lines = []
    header_strats = "  ".join(f"{s:>16s}" for s in strategies)
    sub_header = "  ".join(f"{'p50':>7s} {'p99':>7s}" for _ in strategies)
    lines.append(f"{'Category':<20s}  {header_strats}")
    lines.append(f"{'':20s}  {sub_header}")
    lines.append("-" * (22 + 18 * len(strategies)))

    for group in groups_sorted:
        row = f"{group:<20s}"
        for strategy in strategies:
            metrics = strategy_metrics[strategy]
            durs = sorted(metrics[sid]["duration"]
                          for sid in metrics if pid_to_group.get(sid) == group)
            if durs:
                p50 = np.percentile(durs, 50)
                p99 = np.percentile(durs, 99)
                row += f"  {p50:>7.1f} {p99:>7.1f}"
            else:
                row += f"  {'—':>7s} {'—':>7s}"
        lines.append(row)

    row = f"{'ALL':<20s}"
    for strategy in strategies:
        durs = sorted(m["duration"] for m in strategy_metrics[strategy].values())
        if durs:
            p50 = np.percentile(durs, 50)
            p99 = np.percentile(durs, 99)
            row += f"  {p50:>7.1f} {p99:>7.1f}"
        else:
            row += f"  {'—':>7s} {'—':>7s}"
    lines.append(row)

    print("\n" + "\n".join(lines) + "\n")

    if out_path:
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[compare_reports] Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare session duration CDFs across strategies")
    parser.add_argument("--drr", required=True, help="DRR report directory")
    parser.add_argument("--las", required=True, help="LAS report directory")
    parser.add_argument("--rr",  required=True, help="RR report directory")
    parser.add_argument("-o", "--output-dir", default="comparison_plots",
                        help="Output directory (default: comparison_plots)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics from all strategies
    strategy_metrics = {}
    for name, report_dir in [("DRR", args.drr), ("LAS", args.las), ("RR", args.rr)]:
        metrics = load_strategy_metrics(report_dir)
        strategy_metrics[name] = metrics
        print(f"[compare_reports] {name}: {len(metrics)} sessions with duration data")

    # Use the first strategy's data for quartile grouping (all replay the same sessions)
    first_metrics = next(iter(strategy_metrics.values()))
    req_totals = {sid: m["requests"] for sid, m in first_metrics.items()}
    pid_to_group, group_to_color = build_request_groups(req_totals)

    print(f"[compare_reports] Grouping based on {len(req_totals)} sessions")
    for group in sorted(group_to_color.keys(), key=lambda g: int(g.split("-")[0])):
        count = sum(1 for g in pid_to_group.values() if g == group)
        print(f"[compare_reports]   {group}: {count} sessions")

    print_percentile_table(strategy_metrics, pid_to_group, group_to_color,
                           os.path.join(args.output_dir, "percentile_table.txt"))

    plot_duration_cdf(strategy_metrics, pid_to_group, group_to_color,
                      os.path.join(args.output_dir, "program_duration_cdf_comparison.png"))
    plot_duration_cdf_per_category(strategy_metrics, pid_to_group, group_to_color,
                                  args.output_dir)
    plot_avg_program_duration(strategy_metrics,
                             os.path.join(args.output_dir, "avg_program_duration.png"))
    plot_avg_request_completion(strategy_metrics,
                               os.path.join(args.output_dir, "avg_request_completion.png"))
    plot_sum_program_duration(strategy_metrics,
                             os.path.join(args.output_dir, "sum_program_duration.png"))

    print(f"[compare_reports] Done — output in {args.output_dir}/")


if __name__ == "__main__":
    main()

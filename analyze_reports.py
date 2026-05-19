#!/usr/bin/env python3
"""
Analyze agentic-trace report folders and produce a program duration scatter plot.

Uses quartile-based grouping by LLM call count (num_events) per session.
Reads per_session_lifecycle_metrics.json from the report directory.

Usage:
    python3 analyze_reports.py reports-two-sessions-sequential
    python3 analyze_reports.py reports-two-sessions-sequential -o /out/
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_session_metrics(report_dir: str) -> List[dict]:
    path = os.path.join(report_dir, "per_session_lifecycle_metrics.json")
    if not os.path.exists(path):
        print(f"[analyze_reports] ERROR: {path} not found. "
              f"Ensure per_session: true is set in the report config.")
        return []
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Color mapping — quartile bins by num_events (LLM call count)
# ---------------------------------------------------------------------------

def _build_request_bins(totals: Dict[int, float]):
    vals = sorted(totals.values())
    n = len(vals)
    if n == 0:
        return {}, {}

    q1, q2, q3 = vals[n // 4], vals[n // 2], vals[3 * n // 4]
    unique_edges = sorted(set([vals[0], q1, q2, q3, vals[-1]]))

    if len(unique_edges) < 3:
        label = f"{int(vals[0])}-{int(vals[-1])} reqs"
        return {pid: label for pid in totals}, {label: plt.cm.tab10.colors[0]}

    bins = []
    for i in range(len(unique_edges) - 1):
        lo, hi = unique_edges[i], unique_edges[i + 1]
        bins.append((lo, hi, f"{int(lo)}-{int(hi)} reqs", i == len(unique_edges) - 2))

    colors = plt.cm.tab10.colors
    group_to_color = {label: colors[i % len(colors)] for i, (_, _, label, _) in enumerate(bins)}

    pid_to_group = {}
    for pid, total in totals.items():
        for lo, hi, label, is_last in bins:
            if (is_last and total >= lo) or (not is_last and lo <= total < hi):
                pid_to_group[pid] = label
                break
        else:
            pid_to_group[pid] = bins[-1][2]

    return pid_to_group, group_to_color


# ---------------------------------------------------------------------------
# Plot: Program duration scatter
# ---------------------------------------------------------------------------

def plot_program_duration(sessions, out_path, pid_to_group, group_to_color, req_totals):
    if not sessions:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    seen_groups: set = set()
    min_start = min(s["start_time"] for s in sessions)

    for i, session in enumerate(sessions):
        x = session["start_time"] - min_start
        y = session["duration_sec"]
        n_reqs = int(req_totals.get(i, 0))

        group = pid_to_group.get(i, "unknown")
        color = group_to_color.get(group, (0.5, 0.5, 0.5))
        label = group if group not in seen_groups else "_nolegend_"
        seen_groups.add(group)
        ax.scatter([x], [y], color=color, s=50, zorder=3, label=label)
        ax.annotate(str(n_reqs), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.8)

    ax.set_xlabel("Session Start Time (s)")
    ax.set_ylabel("Session Duration (s)")
    ax.set_title("Session Duration vs Start Time (from per_session_lifecycle_metrics)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze_reports] Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot session duration from per_session_lifecycle_metrics.json")
    parser.add_argument("report_dir", help="Path to the report directory")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory for plots (default: <report_dir>/report_plots)")
    args = parser.parse_args()

    report_dir = args.report_dir
    if not os.path.isdir(report_dir):
        print(f"[analyze_reports] Not a directory: {report_dir}")
        return

    sessions = load_per_session_metrics(report_dir)
    if not sessions:
        return
    print(f"[analyze_reports] Loaded {len(sessions)} sessions from {report_dir}")

    req_totals = {i: s["num_events"] for i, s in enumerate(sessions)}
    pid_to_group, group_to_color = _build_request_bins(req_totals)

    for group in sorted(group_to_color.keys()):
        count = sum(1 for g in pid_to_group.values() if g == group)
        print(f"[analyze_reports]   {group}: {count} sessions")

    out_dir = args.output_dir or report_dir
    plots_dir = os.path.join(out_dir, "report_plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_program_duration(sessions,
                          os.path.join(plots_dir, "program_duration.png"),
                          pid_to_group, group_to_color, req_totals)

    print(f"[analyze_reports] All plots written to {plots_dir}/")


if __name__ == "__main__":
    main()

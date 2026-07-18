"""Render the persisted pre-v3 benchmark series against Kinegraph v3 targets."""
from __future__ import annotations

import argparse
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    "Faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Context Recall",
    "Answer Correctness",
]
BASELINE = [0.3292, 0.1016, 1.0000, 0.3476, 0.3745]
COMPOSED_PRE_V3 = [0.6000, 0.5659, 0.4500, 0.6000, 0.4082]
TARGETS = [0.7500, 0.6500, 0.9000, 0.6500, 0.6000]


def render(output: Path) -> None:
    count = len(METRICS)
    angles = [index / count * 2 * pi for index in range(count)]
    closed_angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8.5), subplot_kw={"polar": True})
    for values, label, color, linestyle in (
        (BASELINE, "Recorded baseline (pre-v3)", "#ef4444", "-"),
        (COMPOSED_PRE_V3, "Recorded composed (pre-v3)", "#7c3aed", "-"),
        (TARGETS, "v3 target", "#16a34a", "--"),
    ):
        closed_values = values + values[:1]
        ax.plot(closed_angles, closed_values, linewidth=2.2, label=label, color=color, linestyle=linestyle)
        if label != "v3 target":
            ax.fill(closed_angles, closed_values, alpha=0.08, color=color)

    ax.set_xticks(angles)
    ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#64748b")
    ax.set_title("Kinegraph Benchmark Status — Pre-v3 Evidence vs v3 Targets", pad=28, fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=3, frameon=False)
    fig.subplots_adjust(top=0.88, bottom=0.20)
    fig.text(
        0.5,
        0.025,
        "No accepted post-v3 RAGAS run is persisted.",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/spider_graph_ragas_score.png"),
    )
    args = parser.parse_args()
    render(args.output)
    print(f"Spider graph saved to {args.output}")


if __name__ == "__main__":
    main()

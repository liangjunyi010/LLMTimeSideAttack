#!/usr/bin/env python3
# analyze_lat.py  (self-contained)

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


JSON_PATH = Path("gpt2_raw_lat.json")
OUT_DIR   = Path("figs")



def load_records(path: Path) -> pd.DataFrame:
    """Read the raw JSON and explode latency lists into rows."""
    with path.open() as f:
        raw = json.load(f)

    rows = []
    for rec in raw:
        for lat in rec["lats"]:
            rows.append(
                {"round": rec["round"], "expert": rec["expert"], "lat": lat}
            )
    return pd.DataFrame(rows)


def plot_bar(means, out_dir):
    """Per-round mean latency bar chart (small vs big)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    rounds = sorted(means["round"].unique())
    idx = range(len(rounds))

    ax.bar(
        [i - width / 2 for i in idx],
        means.query("expert == 'small'")["mean_lat"],
        width,
        label="small",
    )
    ax.bar(
        [i + width / 2 for i in idx],
        means.query("expert == 'big'")["mean_lat"],
        width,
        label="big",
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_xticks(idx)
    ax.set_xticklabels(rounds)
    ax.set_title("Mean latency per round")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_latency_bar.png", dpi=300)


def plot_hist(df, out_dir):
    """Overall latency histogram for small and big."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 30
    ax.hist(
        df.query("expert == 'small'")["lat"],
        bins=bins,
        alpha=0.6,
        label="small",
    )
    ax.hist(
        df.query("expert == 'big'")["lat"],
        bins=bins,
        alpha=0.6,
        label="big",
    )
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Latency distribution (all rounds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_hist.png", dpi=300)


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")
    OUT_DIR.mkdir(exist_ok=True)

    df = load_records(JSON_PATH)

    means = (
        df.groupby(["round", "expert"])["lat"]
        .agg(mean_lat="mean", std_lat="std", n="count")
        .reset_index()
    )
    print(means)

    plot_bar(means, OUT_DIR)
    plot_hist(df, OUT_DIR)
    print(f"Charts saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

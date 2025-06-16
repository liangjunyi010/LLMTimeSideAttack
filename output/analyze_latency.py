#!/usr/bin/env python
# --------------------------------------------------
# analyze_latency.py  (smart path version)
# --------------------------------------------------
import json, glob, statistics
from pathlib import Path

def load_latencies(fp: Path):
    with fp.open() as f:
        return json.load(f)["latencies_ms"]

def describe(arr):
    return dict(max=max(arr), min=min(arr), mean=statistics.mean(arr))

def main():
    here = Path(__file__).resolve()
    base = here.parent                     # 运行目录
    # 若脚本放在 output/，则 output_dir = base
    # 否则假定 output/ 与脚本同级
    output_dir = base if base.name == "output" else base / "output"

    small_fp = next(output_dir.glob("*_small.json"))
    big_fp   = output_dir / small_fp.name.replace("_small.json", "_big.json")
    if not big_fp.exists():
        raise FileNotFoundError(f"missing {big_fp.name}")

    small_lat = load_latencies(small_fp)
    big_lat   = load_latencies(big_fp)

    stats_small = describe(small_lat)
    stats_big   = describe(big_lat)
    small_max   = stats_small["max"]
    error_rate  = sum(t < small_max for t in big_lat) / len(big_lat) * 100

    print(f"\nAnalyzed files in {output_dir.relative_to(base)}:")
    print(f"  small → {small_fp.name}")
    print(f"  big   → {big_fp.name}\n")
    print(f"[SMALL] max {stats_small['max']:.3f} ms | "
          f"min {stats_small['min']:.3f} | mean {stats_small['mean']:.3f}")
    print(f"[ BIG ] max {stats_big['max']:.3f} ms | "
          f"min {stats_big['min']:.3f} | mean {stats_big['mean']:.3f}")
    print(f"\nError-rate (small_max > big) : {error_rate:.2f}%\n")

if __name__ == "__main__":
    main()

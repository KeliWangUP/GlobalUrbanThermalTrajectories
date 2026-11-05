#!/usr/bin/env python3
"""Compare D-Feb/Jul and C-Apr/Jul city-level SUHI trends."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.paired_month_city_trends import build_paired_month_city_trends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "paired_month_city_trends_config.json",
        help="Path to the paired-month comparison config JSON.",
    )
    args = parser.parse_args()
    build_paired_month_city_trends(args.config)


if __name__ == "__main__":
    main()

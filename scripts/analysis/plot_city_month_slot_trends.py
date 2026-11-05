#!/usr/bin/env python3
"""Plot climate-specific city/month/slot trend distributions."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.city_month_slot_trend_figures import (
    build_city_month_slot_trend_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "city_month_slot_trend_figures_config.json",
        help="Path to the trend-figure configuration JSON.",
    )
    args = parser.parse_args()
    build_city_month_slot_trend_figures(args.config)


if __name__ == "__main__":
    main()

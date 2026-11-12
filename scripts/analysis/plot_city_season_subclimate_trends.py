#!/usr/bin/env python3
"""Plot seasonal trends for subclasses within a selected Köppen climate."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.city_season_subclimate_trend_figures import (
    build_city_season_subclimate_trend_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "city_season_subclimate_trend_figures_config.json",
        help="Path to the subclimate trend-figure config JSON.",
    )
    args = parser.parse_args()
    build_city_season_subclimate_trend_figures(args.config)


if __name__ == "__main__":
    main()

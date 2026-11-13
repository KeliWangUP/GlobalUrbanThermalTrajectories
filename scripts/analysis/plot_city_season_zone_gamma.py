#!/usr/bin/env python3
"""Plot seasonal gamma by urban pixel zone and Köppen subclass."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.city_season_zone_gamma_figures import (
    build_city_season_zone_gamma_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "city_season_zone_gamma_figures_config.json",
        help="Path to the seasonal zone gamma figure config JSON.",
    )
    args = parser.parse_args()
    build_city_season_zone_gamma_figures(args.config)


if __name__ == "__main__":
    main()

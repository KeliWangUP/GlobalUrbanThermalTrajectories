#!/usr/bin/env python3
"""Fit paired MODIS and ERA5 seasonal trends and common-year contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.city_season_era5_trends import build_city_season_era5_trends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "city_season_era5_trends_config.json",
        help="Path to the paired ERA5 and MODIS trend configuration.",
    )
    args = parser.parse_args()
    build_city_season_era5_trends(args.config)


if __name__ == "__main__":
    main()

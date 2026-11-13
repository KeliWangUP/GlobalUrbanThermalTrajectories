#!/usr/bin/env python3
"""Build seasonal trends and hemisphere-aware gamma for pixel urban zones."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.city_season_zone_trends import build_city_season_zone_trends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "city_season_zone_trends_config.json",
        help="Path to the seasonal zone trend configuration JSON.",
    )
    args = parser.parse_args()
    build_city_season_zone_trends(args.config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build city-level summer-minus-winter seasonal trend contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.city_seasonal_trend_contrasts import (
    build_city_seasonal_trend_contrasts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "city_seasonal_trend_contrasts_config.json",
        help="Path to the seasonal trend contrast config JSON.",
    )
    args = parser.parse_args()
    build_city_seasonal_trend_contrasts(args.config)


if __name__ == "__main__":
    main()

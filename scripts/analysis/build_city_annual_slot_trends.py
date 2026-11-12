#!/usr/bin/env python3
"""Build city/slot trends from annual SUHI data."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.city_annual_slot_trends import build_city_annual_slot_trends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "city_annual_slot_trends_config.json",
        help="Path to the annual trend configuration JSON.",
    )
    args = parser.parse_args()
    build_city_annual_slot_trends(args.config)


if __name__ == "__main__":
    main()

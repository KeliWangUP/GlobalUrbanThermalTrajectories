#!/usr/bin/env python3
"""Build city/month/slot trend slopes from the monthly SUHI table."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.city_month_slot_trends import build_city_month_slot_trends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "city_month_slot_trends_config.json",
        help="Path to the city-month trend configuration JSON.",
    )
    args = parser.parse_args()
    build_city_month_slot_trends(args.config)


if __name__ == "__main__":
    main()

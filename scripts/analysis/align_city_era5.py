#!/usr/bin/env python3
"""Align monthly ERA5-Land temperatures with monthly, seasonal and annual SUHI."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.city_era5_alignment import align_city_era5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "city_era5_alignment_config.json",
        help="Path to the ERA5 alignment configuration JSON.",
    )
    args = parser.parse_args()
    align_city_era5(args.config)


if __name__ == "__main__":
    main()

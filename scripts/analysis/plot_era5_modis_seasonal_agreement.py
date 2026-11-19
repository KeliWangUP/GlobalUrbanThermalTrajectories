#!/usr/bin/env python3
"""Plot regional MODIS and ERA5 warm-minus-cold trend agreement."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.era5_modis_agreement_figures import build_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "era5_modis_agreement_figures_config.json",
    )
    args = parser.parse_args()
    build_figures(args.config)


if __name__ == "__main__":
    main()

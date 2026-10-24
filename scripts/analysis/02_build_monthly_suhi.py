import re
from pathlib import Path

import pandas as pd

from src.build_suhi import (
    KEY_COLUMNS,
    LST_SLOTS,
    aggregate_annual_suhi,
    aggregate_seasonal_suhi,
    build_monthly_suhi,
    write_parquet_atomically
)
from src.config import load_analysis_config


YEAR_PATTERN = re.compile(r"_(\d{4})\.parquet$", re.IGNORECASE)


def index_yearly_parquets(folder: Path) -> dict[int, Path]:
    paths = sorted(folder.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No yearly parquet files found in {folder}")

    yearly_paths = {}
    for path in paths:
        match = YEAR_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Parquet filename does not end with a year: {path}")
        year = int(match.group(1))
        if year in yearly_paths:
            raise ValueError(
                f"Multiple parquet files found for {year} in {folder}."
            )
        yearly_paths[year] = path
    return yearly_paths


def main() -> None:
    config = load_analysis_config()
    urban_files = index_yearly_parquets(config.urban_parquet_dir)
    rural_files = index_yearly_parquets(config.rural_parquet_dir)
    if set(urban_files) != set(rural_files):
        raise ValueError(
            "Urban and rural parquet years do not match. "
            f"Urban-only: {sorted(set(urban_files) - set(rural_files))}; "
            f"rural-only: {sorted(set(rural_files) - set(urban_files))}."
        )

    yearly_panels = []
    for year in sorted(urban_files):
        print(f"[JOIN] Building monthly SUHI for {year}", flush=True)
        urban = pd.read_parquet(urban_files[year])
        rural = pd.read_parquet(rural_files[year])
        monthly = build_monthly_suhi(urban, rural, year)
        yearly_panels.append(monthly)
        print(
            f"  rows={len(monthly):,}; "
            f"cities={monthly['ID_UC_G0'].nunique():,}",
            flush=True
        )

    monthly_panel = pd.concat(yearly_panels, ignore_index=True)
    if monthly_panel.duplicated(KEY_COLUMNS).any():
        raise ValueError("Monthly SUHI panel contains duplicate keys.")
    monthly_panel = monthly_panel.sort_values(
        KEY_COLUMNS,
        kind="stable"
    ).reset_index(drop=True)

    seasonal_panel = aggregate_seasonal_suhi(monthly_panel)
    annual_panel = aggregate_annual_suhi(monthly_panel)

    outputs = (
        (config.monthly_suhi_file, monthly_panel),
        (config.seasonal_suhi_file, seasonal_panel),
        (config.annual_suhi_file, annual_panel)
    )
    for output_path, frame in outputs:
        write_parquet_atomically(
            frame,
            output_path,
            config.parquet_compression
        )
        print(
            f"[OUTPUT] {output_path}: {len(frame):,} rows; "
            f"columns={len(frame.columns)}",
            flush=True
        )

    print(
        f"[DONE] SUHI slots: {', '.join(LST_SLOTS)}; "
        "seasonal and annual means use equal monthly weights.",
        flush=True
    )


if __name__ == "__main__":
    main()
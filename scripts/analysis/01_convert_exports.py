import re
from pathlib import Path

import pandas as pd

from src.build_suhi import write_parquet_atomically
from src.config import load_analysis_config


KEY_COLUMNS = ["ID_UC_G0", "year", "month"]
LST_SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")
LST_STATISTICS = ("mean", "median", "stdDev", "count")
QC_COLUMNS = [
    f"{slot}_{metric}_mean"
    for slot in LST_SLOTS
    for metric in ("clear_days", "n8day")
]
LST_COLUMNS = [
    f"{slot}_{statistic}"
    for slot in LST_SLOTS
    for statistic in LST_STATISTICS
]
EXPECTED_COLUMNS = KEY_COLUMNS + LST_COLUMNS + QC_COLUMNS
COUNT_COLUMNS = [f"{slot}_count" for slot in LST_SLOTS]
YEAR_PATTERN = re.compile(r"_(\d{4})\.csv$", re.IGNORECASE)


def get_year(path: Path) -> int:
    match = YEAR_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"CSV filename does not end with a year: {path}")
    return int(match.group(1))


def get_yearly_csvs(folder: Path) -> dict[int, Path]:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    yearly_files = {}
    for path in files:
        year = get_year(path)
        if year in yearly_files:
            raise ValueError(
                f"Multiple CSV files found for {year} in {folder}: "
                f"{yearly_files[year].name}, {path.name}"
            )
        yearly_files[year] = path
    return yearly_files


def validate_clean_export(frame: pd.DataFrame, path: Path, year: int) -> None:
    if frame.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {path}. "
            f"Expected {EXPECTED_COLUMNS}; found {frame.columns.tolist()}."
        )
    if frame.empty:
        raise ValueError(f"CSV contains no records: {path}")
    if frame["year"].nunique() != 1 or int(frame["year"].iloc[0]) != year:
        raise ValueError(
            f"Year column does not match filename year {year}: {path}"
        )
    if not frame["month"].between(1, 12).all():
        raise ValueError(f"Month values must be in 1..12: {path}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Duplicate city/year/month keys found: {path}")

    months_by_city = frame.groupby("ID_UC_G0").agg(
        row_count=("month", "size"),
        unique_months=("month", "nunique")
    )
    incomplete = months_by_city[
        (months_by_city["row_count"] != 12)
        | (months_by_city["unique_months"] != 12)
    ]
    if not incomplete.empty:
        examples = incomplete.head(5).to_dict("index")
        raise ValueError(
            f"Each city must have exactly 12 monthly records in {path}; "
            f"examples: {examples}"
        )


def convert_csv(
    path: Path,
    output_path: Path,
    year: int,
    compression: str
) -> None:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={
            "ID_UC_G0": "int32",
            "year": "int16",
            "month": "int8"
        }
    )
    for column in EXPECTED_COLUMNS[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    for column in COUNT_COLUMNS:
        values = frame[column].dropna()
        if not values.mod(1).eq(0).all():
            raise ValueError(f"Expected integer pixel counts in {path}:{column}")
        frame[column] = frame[column].astype("Int32")

    validate_clean_export(frame, path, year)
    write_parquet_atomically(frame, output_path, compression)

    print(
        f"[PARQUET] {path.name}: {len(frame):,} rows, "
        f"{frame['ID_UC_G0'].nunique():,} cities -> {output_path}",
        flush=True
    )


def main() -> None:
    config = load_analysis_config()
    datasets = {
        "urban": (config.urban_raw_dir, config.urban_parquet_dir),
        "rural": (config.rural_raw_dir, config.rural_parquet_dir)
    }
    yearly_files = {
        name: get_yearly_csvs(raw_folder)
        for name, (raw_folder, _) in datasets.items()
    }
    urban_years = set(yearly_files["urban"])
    rural_years = set(yearly_files["rural"])
    if urban_years != rural_years:
        raise ValueError(
            "Urban and rural exports must cover the same years. "
            f"Urban-only: {sorted(urban_years - rural_years)}; "
            f"rural-only: {sorted(rural_years - urban_years)}."
        )

    years = sorted(urban_years)
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError(f"Export years are not continuous: {years}")

    print(
        f"[INFO] Converting {len(years)} annual CSV pairs "
        f"({years[0]}-{years[-1]}).",
        flush=True
    )
    for dataset_name, source_files in yearly_files.items():
        output_folder = datasets[dataset_name][1]
        output_folder.mkdir(parents=True, exist_ok=True)
        for year in years:
            source_path = source_files[year]
            output_path = output_folder / f"{dataset_name}_{year}.parquet"
            convert_csv(
                source_path,
                output_path,
                year,
                config.parquet_compression
            )


if __name__ == "__main__":
    main()
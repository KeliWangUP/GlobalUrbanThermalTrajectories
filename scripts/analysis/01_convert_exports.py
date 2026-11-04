from src.city_lst_csv import (
    COUNT_COLUMNS as COUNT_COLUMNS,
    EXPECTED_COLUMNS as EXPECTED_COLUMNS,
    KEY_COLUMNS as KEY_COLUMNS,
    LST_COLUMNS as LST_COLUMNS,
    LST_SLOTS as LST_SLOTS,
    LST_STATISTICS as LST_STATISTICS,
    QC_COLUMNS as QC_COLUMNS,
    YEAR_PATTERN as YEAR_PATTERN,
    convert_csv,
    get_year as get_year,
    get_yearly_csvs,
    load_clean_export as load_clean_export,
    validate_clean_export as validate_clean_export
)
from src.config import load_analysis_config


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

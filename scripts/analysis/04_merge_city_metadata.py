import pandas as pd

from src.build_suhi import write_parquet_atomically
from src.city_metadata import merge_city_metadata
from src.config import load_analysis_config


def main() -> None:
    config = load_analysis_config()
    if not config.city_metadata_file.is_file():
        raise FileNotFoundError(
            f"City metadata Parquet not found: {config.city_metadata_file}. "
            "Run 02_build_monthly_suhi.py to generate it."
        )

    outputs = (
        config.monthly_suhi_file,
        config.seasonal_suhi_file,
        config.annual_suhi_file
    )
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot merge city metadata; SUHI Parquet files are missing: "
            f"{missing}. Run 02_build_monthly_suhi.py first."
        )

    metadata = pd.read_parquet(config.city_metadata_file)
    enriched = []
    for path in outputs:
        panel = pd.read_parquet(path)
        enriched.append((path, merge_city_metadata(panel, metadata)))

    for path, panel in enriched:
        write_parquet_atomically(
            panel,
            path,
            config.parquet_compression
        )
        print(
            f"[OUTPUT] {path}: {len(panel):,} rows; "
            f"{panel['centroid_lat'].notna().sum():,} rows with centroid "
            f"latitude.",
            flush=True
        )


if __name__ == "__main__":
    main()

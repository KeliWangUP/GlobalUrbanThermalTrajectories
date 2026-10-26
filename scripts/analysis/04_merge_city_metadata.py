import pandas as pd

from src.build_suhi import write_parquet_atomically
from src.city_metadata import build_city_metadata, merge_city_metadata
from src.config import load_analysis_config


def main() -> None:
    config = load_analysis_config()
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

    metadata, repaired_geometry_count = build_city_metadata(
        config.city_bounds_shapefile,
        config.koppen_geiger_raster,
        config.cz_macro_groups
    )
    write_parquet_atomically(
        metadata,
        config.city_metadata_file,
        config.parquet_compression
    )
    print(
        f"[CITY META] Refreshed {len(metadata):,} cities; "
        f"{repaired_geometry_count:,} invalid geometries repaired; "
        f"{metadata['koppen_geiger_macro_group'].notna().sum():,} cities "
        f"assigned a climate class and macro group; "
        f"{metadata['koppen_geiger_class'].isna().sum():,} no-data centroids.",
        flush=True
    )

    enriched = []
    for path in outputs:
        panel = pd.read_parquet(path)
        enriched.append((
            path,
            merge_city_metadata(panel, metadata, config.cz_macro_groups)
        ))

    for path, panel in enriched:
        write_parquet_atomically(
            panel,
            path,
            config.parquet_compression
        )
        print(
            f"[OUTPUT] {path}: {len(panel):,} rows; "
            f"{panel['centroid_lat'].notna().sum():,} rows with centroid "
            f"latitude; "
            f"{panel['koppen_geiger_class'].notna().sum():,} rows with "
            "climate class.",
            flush=True
        )


if __name__ == "__main__":
    main()

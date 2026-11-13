from src.build_suhi import write_parquet_atomically
from src.city_metadata import build_city_metadata
from src.config import load_analysis_config


def main() -> None:
    config = load_analysis_config()
    metadata, repaired_geometry_count = build_city_metadata(
        config.city_bounds_shapefile,
        config.koppen_geiger_raster,
        config.cz_macro_groups,
        built_development_csv_path=config.ghsl_built_development_csv
    )
    classified_count = int(metadata["koppen_geiger_class"].notna().sum())
    no_data_count = len(metadata) - classified_count
    write_parquet_atomically(
        metadata,
        config.city_metadata_file,
        config.parquet_compression
    )
    print(
        f"[OUTPUT] {config.city_metadata_file}: {len(metadata):,} cities; "
        f"{metadata['centroid_lon'].notna().sum():,} centroids with longitude; "
        f"{metadata['centroid_lat'].notna().sum():,} centroids with latitude; "
        f"{metadata['city_area_km2'].notna().sum():,} cities with area; "
        f"{metadata['country'].notna().sum():,} cities with country; "
        f"{metadata['GC_DEV_USR'].notna().sum():,} cities with GHSL region; "
        f"{metadata['built_surface_slope_m2_per_year'].notna().sum():,} "
        "cities with built-development metrics; "
        f"{classified_count:,} cities assigned a climate class and macro "
        "group; "
        f"{no_data_count:,} raster no-data centroids; "
        f"{repaired_geometry_count:,} invalid geometries repaired.",
        flush=True
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize stable/development pixel counts for each city in a Gold build."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError as exc:
    raise SystemExit(
        "Install project dependencies first: pip install -r requirements.txt"
    ) from exc


REQUIRED_COLUMNS = {
    "ID_UC_G0",
    "pixel_id",
    "urban_zone",
    "water_eligible",
}


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _resolve_path(config_path: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config 'paths.{key}' must be a non-empty path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _load_config(
    config_path: Path,
) -> tuple[dict[str, Any], Path, Path, Path, Path, dict[str, str]]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise ValueError("Pixel-count config must have a JSON object and 'paths'.")
    paths = config["paths"]
    classification_file = _resolve_path(
        config_path, paths.get("pixel_classification"), "pixel_classification"
    )
    gold_manifest = _resolve_path(
        config_path, paths.get("gold_manifest"), "gold_manifest"
    )
    city_metadata = _resolve_path(
        config_path, paths.get("city_metadata"), "city_metadata"
    )
    output_csv = _resolve_path(config_path, paths.get("output_csv"), "output_csv")
    columns = config.get("columns")
    required_metadata = (
        "city_id",
        "longitude",
        "latitude",
        "city_area",
        "country",
        "ghsl_region",
        "koppen_code",
        "koppen_class",
        "koppen_macro_group",
    )
    if not isinstance(columns, dict) or any(
        not isinstance(columns.get(name), str) or not columns[name].strip()
        for name in required_metadata
    ):
        raise ValueError(
            "Config 'columns' must map all required city metadata fields."
        )
    metadata_columns = {name: columns[name].strip() for name in required_metadata}
    return (
        config,
        classification_file,
        gold_manifest,
        city_metadata,
        output_csv,
        metadata_columns,
    )


def build_city_pixel_zone_counts(config_path: Path) -> Path:
    (
        _,
        source,
        gold_manifest_path,
        metadata_path,
        output_csv,
        metadata_columns,
    ) = _load_config(config_path)
    if not source.is_file():
        raise FileNotFoundError(f"Pixel classification table not found: {source}")
    if not gold_manifest_path.is_file():
        raise FileNotFoundError(f"Gold manifest not found: {gold_manifest_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"City metadata table not found: {metadata_path}")

    gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8"))
    classification = gold_manifest.get("urban_classification")
    if not isinstance(classification, dict):
        raise ValueError("Gold manifest has no urban_classification metadata.")
    stable_label = classification.get("stable_label")
    development_label = classification.get("development_label")
    if (
        not isinstance(stable_label, str)
        or not stable_label
        or not isinstance(development_label, str)
        or not development_label
        or stable_label == development_label
    ):
        raise ValueError("Gold manifest has invalid urban classification labels.")
    baseline_year = classification.get("baseline_year")
    threshold = classification.get("stable_isa_threshold")
    city_count_expected = gold_manifest.get("city_count")
    if (
        type(baseline_year) is not int
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
        or type(city_count_expected) is not int
        or city_count_expected < 1
    ):
        raise ValueError("Gold manifest has invalid classification/count metadata.")

    connection = duckdb.connect(":memory:")
    temporary_path: Path | None = None
    try:
        description = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [source.as_posix()]
        ).fetchall()
        available = {row[0] for row in description}
        missing = REQUIRED_COLUMNS - available
        if missing:
            raise ValueError(
                f"Pixel classification table is missing fields: {sorted(missing)}."
            )
        water_type = next(row[1] for row in description if row[0] == "water_eligible")
        if water_type != "BOOLEAN":
            raise TypeError("water_eligible must have BOOLEAN type.")

        metadata_description = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [metadata_path.as_posix()]
        ).fetchall()
        metadata_available = {row[0] for row in metadata_description}
        missing_metadata = set(metadata_columns.values()) - metadata_available
        if missing_metadata:
            raise ValueError(
                f"City metadata table is missing fields: "
                f"{sorted(missing_metadata)}."
            )

        metadata_query = f"""
            SELECT
                CAST({_identifier(metadata_columns['city_id'])} AS BIGINT)
                    AS ID_UC_G0,
                CAST({_identifier(metadata_columns['longitude'])} AS DOUBLE)
                    AS centroid_lon,
                CAST({_identifier(metadata_columns['latitude'])} AS DOUBLE)
                    AS centroid_lat,
                CAST({_identifier(metadata_columns['city_area'])} AS DOUBLE)
                    AS city_area_km2,
                CAST({_identifier(metadata_columns['country'])} AS VARCHAR)
                    AS country,
                CAST({_identifier(metadata_columns['ghsl_region'])} AS VARCHAR)
                    AS GC_DEV_USR,
                CAST({_identifier(metadata_columns['koppen_code'])} AS INTEGER)
                    AS koppen_geiger_code,
                CAST({_identifier(metadata_columns['koppen_class'])} AS VARCHAR)
                    AS koppen_geiger_class,
                CAST({_identifier(metadata_columns['koppen_macro_group'])} AS VARCHAR)
                    AS koppen_geiger_macro_group
            FROM read_parquet({_literal(metadata_path.as_posix())})
        """
        duplicate_metadata = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT ID_UC_G0
                FROM ({metadata_query}) metadata
                GROUP BY ID_UC_G0
                HAVING ID_UC_G0 IS NULL OR COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        if duplicate_metadata:
            raise ValueError(
                "City metadata has null or duplicate city IDs."
            )
        invalid_metadata = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM ({metadata_query}) metadata
            WHERE centroid_lon IS NULL OR NOT isfinite(centroid_lon)
               OR centroid_lon < -180 OR centroid_lon > 180
               OR centroid_lat IS NULL OR NOT isfinite(centroid_lat)
               OR centroid_lat < -90 OR centroid_lat > 90
               OR city_area_km2 IS NULL OR NOT isfinite(city_area_km2)
               OR city_area_km2 < 0
            """
        ).fetchone()[0]
        if invalid_metadata:
            raise ValueError(
                f"City metadata contains {invalid_metadata:,} invalid location rows."
            )

        invalid = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet({_literal(source.as_posix())})
            WHERE ID_UC_G0 IS NULL
               OR pixel_id IS NULL
               OR urban_zone IS NULL
               OR urban_zone NOT IN (
                   {_literal(stable_label)}, {_literal(development_label)}
               )
               OR water_eligible IS NULL
            """
        ).fetchone()[0]
        if invalid:
            raise ValueError(
                f"Pixel classification table contains {invalid:,} invalid rows."
            )

        duplicates = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT ID_UC_G0, pixel_id
                FROM read_parquet({_literal(source.as_posix())})
                GROUP BY ID_UC_G0, pixel_id
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        if duplicates:
            raise ValueError(
                f"Pixel classification table contains {duplicates:,} "
                "duplicate city/pixel keys."
            )

        present_labels = {
            row[0]
            for row in connection.execute(
                f"""
                SELECT DISTINCT urban_zone
                FROM read_parquet({_literal(source.as_posix())})
                """
            ).fetchall()
        }
        if present_labels != {stable_label, development_label}:
            raise ValueError(
                "Pixel labels differ from the Gold manifest: "
                f"found={sorted(present_labels)}, "
                f"expected={sorted((stable_label, development_label))}."
            )

        counts_query = f"""
            SELECT
                ID_UC_G0,
                COUNT(*) AS total_pixel_count,
                COUNT(*) FILTER (
                    WHERE urban_zone = {_literal(stable_label)}
                ) AS stable_pixel_count,
                COUNT(*) FILTER (
                    WHERE urban_zone = {_literal(development_label)}
                ) AS development_pixel_count,
                COUNT(*) FILTER (WHERE water_eligible)
                    AS water_eligible_pixel_count,
                COUNT(*) FILTER (
                    WHERE urban_zone = {_literal(stable_label)}
                      AND water_eligible
                ) AS stable_water_eligible_pixel_count,
                COUNT(*) FILTER (
                    WHERE urban_zone = {_literal(development_label)}
                      AND water_eligible
                ) AS development_water_eligible_pixel_count,
                COUNT(*) FILTER (
                    WHERE urban_zone = {_literal(stable_label)}
                      AND NOT water_eligible
                ) AS stable_water_excluded_pixel_count,
                COUNT(*) FILTER (
                    WHERE urban_zone = {_literal(development_label)}
                      AND NOT water_eligible
                ) AS development_water_excluded_pixel_count,
                {baseline_year} AS isa_baseline_year,
                {float(threshold)} AS stable_isa_threshold
            FROM read_parquet({_literal(source.as_posix())})
            GROUP BY ID_UC_G0
        """
        city_count, invalid_totals = connection.execute(
            f"""
            SELECT COUNT(*),
                   COUNT(*) FILTER (
                       WHERE total_pixel_count !=
                                 stable_pixel_count + development_pixel_count
                          OR water_eligible_pixel_count !=
                                 stable_water_eligible_pixel_count
                                   + development_water_eligible_pixel_count
                   )
            FROM ({counts_query}) counts
            """
        ).fetchone()
        if city_count != city_count_expected:
            raise ValueError(
                f"Pixel count table has {city_count} cities; expected "
                f"{city_count_expected} from the Gold manifest."
            )
        if invalid_totals:
            raise ValueError(
                f"Pixel count aggregation has {invalid_totals} inconsistent cities."
            )
        missing_metadata_cities = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM ({counts_query}) counts
            LEFT JOIN ({metadata_query}) metadata USING (ID_UC_G0)
            WHERE metadata.ID_UC_G0 IS NULL
            """
        ).fetchone()[0]
        if missing_metadata_cities:
            raise ValueError(
                f"City metadata is missing {missing_metadata_cities:,} cities "
                "from the pixel classification table."
            )

        query = f"""
            SELECT counts.*,
                   metadata.centroid_lon,
                   metadata.centroid_lat,
                   CASE
                       WHEN metadata.centroid_lat > 0 THEN 'north'
                       WHEN metadata.centroid_lat < 0 THEN 'south'
                       ELSE 'equator'
                   END AS hemisphere,
                   metadata.city_area_km2,
                   metadata.country,
                   metadata.GC_DEV_USR,
                   metadata.koppen_geiger_code,
                   metadata.koppen_geiger_class,
                   LEFT(metadata.koppen_geiger_class, 1) AS climate_major,
                   metadata.koppen_geiger_macro_group
            FROM ({counts_query}) counts
            JOIN ({metadata_query}) metadata USING (ID_UC_G0)
            ORDER BY counts.ID_UC_G0
        """

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{output_csv.stem}_",
            suffix=".csv",
            dir=output_csv.parent,
        )
        os.close(handle)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        connection.execute(
            f"COPY ({query}) TO {_literal(temporary_path.as_posix())} "
            "(FORMAT CSV, HEADER, DELIMITER ',')"
        )
        os.replace(temporary_path, output_csv)
        temporary_path = None
        totals = connection.execute(
            f"""
            SELECT SUM(total_pixel_count), SUM(stable_pixel_count),
                   SUM(development_pixel_count), SUM(water_eligible_pixel_count)
            FROM ({counts_query}) counts
            """
        ).fetchone()
        print(
            f"[DONE] {city_count:,} city counts written to {output_csv}; "
            f"total={totals[0]:,}, stable={totals[1]:,}, "
            f"development={totals[2]:,}, water-eligible={totals[3]:,}.",
            flush=True,
        )
        return output_csv
    finally:
        connection.close()
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("city_pixel_zone_counts_config.json"),
        help="Path to the pixel zone count config JSON.",
    )
    args = parser.parse_args()
    build_city_pixel_zone_counts(args.config)


if __name__ == "__main__":
    main()

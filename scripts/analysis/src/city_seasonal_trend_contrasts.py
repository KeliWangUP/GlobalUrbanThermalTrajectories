"""Build hemisphere-aware summer-minus-winter city trend contrasts."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


SLOTS = (
    "terra_day",
    "terra_night",
    "aqua_day",
    "aqua_night",
)
METRICS = ("suhi", "urban", "rural")
SEASONS = ("DJF", "MAM", "JJA", "SON")
REQUIRED_COLUMNS = {
    "ID_UC_G0",
    "climate",
    "centroid_lon",
    "centroid_lat",
    "city_area_km2",
    "country",
    "GC_DEV_USR",
    "built_surface_2000_m2",
    "built_surface_slope_m2_per_year",
    "built_volume_total_2000_m3",
    "built_volume_total_slope_m3_per_year",
    "hemisphere",
    "season",
    "slot",
    "beta_suhi",
    "beta_urban",
    "beta_rural",
    "n_valid_years",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quote_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Config must contain a 'paths' object.")
    for key in ("seasonal_trend_file", "output_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config 'paths.{key}' must be a non-empty path.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths[key] = path.resolve().as_posix()

    tolerance = config.get("decomposition_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError(
            "Config 'decomposition_tolerance' must be a finite "
            "non-negative number."
        )
    config["decomposition_tolerance"] = float(tolerance)
    return config


def _validate_source(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    tolerance: float,
) -> dict[str, int | float]:
    parquet_path = _quote_string(source.as_posix())
    description = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({parquet_path})"
    ).fetchall()
    available = {row[0] for row in description}
    missing = REQUIRED_COLUMNS - available
    if missing:
        raise ValueError(
            f"Seasonal trend table is missing fields: {sorted(missing)}."
        )

    row_count = connection.execute(
        f"SELECT COUNT(*) FROM read_parquet({parquet_path})"
    ).fetchone()[0]
    if not row_count:
        raise ValueError("Seasonal trend table contains no rows.")

    duplicates = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT ID_UC_G0, season, slot
            FROM read_parquet({parquet_path})
            GROUP BY ID_UC_G0, season, slot
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicates:
        raise ValueError(
            "Seasonal trend table contains duplicate city/season/slot keys."
        )

    invalid_rows = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet({parquet_path})
        WHERE ID_UC_G0 IS NULL
           OR season IS NULL OR season NOT IN ({", ".join(_quote_string(s) for s in SEASONS)})
           OR slot IS NULL OR slot NOT IN ({", ".join(_quote_string(s) for s in SLOTS)})
           OR hemisphere IS NULL
           OR LOWER(TRIM(hemisphere)) NOT IN ('north', 'south', 'equator')
           OR centroid_lon IS NULL
           OR NOT isfinite(CAST(centroid_lon AS DOUBLE))
           OR ABS(CAST(centroid_lon AS DOUBLE)) > 180
           OR centroid_lat IS NULL OR NOT isfinite(CAST(centroid_lat AS DOUBLE))
           OR ABS(CAST(centroid_lat AS DOUBLE)) > 90
           OR city_area_km2 IS NULL
           OR NOT isfinite(CAST(city_area_km2 AS DOUBLE))
           OR CAST(city_area_km2 AS DOUBLE) <= 0
           OR country IS NULL OR TRIM(CAST(country AS VARCHAR)) = ''
           OR GC_DEV_USR IS NULL OR TRIM(CAST(GC_DEV_USR AS VARCHAR)) = ''
           OR built_surface_2000_m2 IS NULL
           OR NOT isfinite(CAST(built_surface_2000_m2 AS DOUBLE))
           OR CAST(built_surface_2000_m2 AS DOUBLE) < 0
           OR built_surface_slope_m2_per_year IS NULL
           OR NOT isfinite(CAST(built_surface_slope_m2_per_year AS DOUBLE))
           OR built_volume_total_2000_m3 IS NULL
           OR NOT isfinite(CAST(built_volume_total_2000_m3 AS DOUBLE))
           OR CAST(built_volume_total_2000_m3 AS DOUBLE) < 0
           OR built_volume_total_slope_m3_per_year IS NULL
           OR NOT isfinite(
               CAST(built_volume_total_slope_m3_per_year AS DOUBLE)
           )
           OR n_valid_years IS NULL OR n_valid_years < 2
           OR beta_suhi IS NULL OR NOT isfinite(CAST(beta_suhi AS DOUBLE))
           OR beta_urban IS NULL OR NOT isfinite(CAST(beta_urban AS DOUBLE))
           OR beta_rural IS NULL OR NOT isfinite(CAST(beta_rural AS DOUBLE))
        """
    ).fetchone()[0]
    if invalid_rows:
        raise ValueError(
            f"Seasonal trend table contains {invalid_rows:,} rows with invalid "
            "season, slot, city metadata, GHSL metadata, valid-year count, "
            "or beta values."
        )

    inconsistent_metadata = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT ID_UC_G0
            FROM read_parquet({parquet_path})
            GROUP BY ID_UC_G0
            HAVING COUNT(DISTINCT centroid_lon) > 1
                OR COUNT(DISTINCT centroid_lat) > 1
                OR COUNT(DISTINCT city_area_km2) > 1
                OR COUNT(DISTINCT country) > 1
                OR COUNT(DISTINCT GC_DEV_USR) > 1
                OR COUNT(DISTINCT built_surface_2000_m2) > 1
                OR COUNT(DISTINCT built_surface_slope_m2_per_year) > 1
                OR COUNT(DISTINCT built_volume_total_2000_m3) > 1
                OR COUNT(DISTINCT built_volume_total_slope_m3_per_year) > 1
                OR COUNT(DISTINCT LOWER(TRIM(hemisphere))) > 1
                OR COUNT(DISTINCT climate) > 1
        )
        """
    ).fetchone()[0]
    if inconsistent_metadata:
        raise ValueError(
            "Seasonal trend table has inconsistent city location, area, "
            "country, region, climate, hemisphere, or GHSL metadata."
        )

    invalid_hemisphere_latitude = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet({parquet_path})
        WHERE (LOWER(TRIM(hemisphere)) = 'north' AND centroid_lat <= 0)
           OR (LOWER(TRIM(hemisphere)) = 'south' AND centroid_lat >= 0)
           OR (LOWER(TRIM(hemisphere)) = 'equator' AND centroid_lat <> 0)
        """
    ).fetchone()[0]
    if invalid_hemisphere_latitude:
        raise ValueError(
            "Seasonal trend hemisphere labels do not match centroid latitude."
        )

    max_source_error = connection.execute(
        f"""
        SELECT COALESCE(
            MAX(ABS(
                CAST(beta_suhi AS DOUBLE)
                - (CAST(beta_urban AS DOUBLE) - CAST(beta_rural AS DOUBLE))
            )),
            0
        )
        FROM read_parquet({parquet_path})
        """
    ).fetchone()[0]
    if max_source_error > tolerance:
        raise ValueError(
            "Input city-level trend decomposition failed: "
            f"max error={max_source_error:.12g}, tolerance={tolerance:.12g}."
        )

    return {
        "source_rows": int(row_count),
        "source_cities": int(
            connection.execute(
                f"SELECT COUNT(DISTINCT ID_UC_G0) "
                f"FROM read_parquet({parquet_path})"
            ).fetchone()[0]
        ),
        "equatorial_cities": int(
            connection.execute(
                f"""
                SELECT COUNT(DISTINCT ID_UC_G0)
                FROM read_parquet({parquet_path})
                WHERE LOWER(TRIM(hemisphere)) = 'equator'
                """
            ).fetchone()[0]
        ),
        "north_south_city_slot_candidates": int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT ID_UC_G0, slot
                    FROM read_parquet({parquet_path})
                    WHERE LOWER(TRIM(hemisphere)) IN ('north', 'south')
                )
                """
            ).fetchone()[0]
        ),
        "max_input_decomposition_error": float(max_source_error),
    }


def _contrast_query(source: Path) -> str:
    source_path = _quote_string(source.as_posix())
    return f"""
        WITH trends AS (
            SELECT
                ID_UC_G0,
                climate,
                CAST(centroid_lon AS DOUBLE) AS centroid_lon,
                CAST(centroid_lat AS DOUBLE) AS centroid_lat,
                CAST(city_area_km2 AS DOUBLE) AS city_area_km2,
                CAST(country AS VARCHAR) AS country,
                CAST(GC_DEV_USR AS VARCHAR) AS GC_DEV_USR,
                CAST(built_surface_2000_m2 AS DOUBLE)
                    AS built_surface_2000_m2,
                CAST(built_surface_slope_m2_per_year AS DOUBLE)
                    AS built_surface_slope_m2_per_year,
                CAST(built_volume_total_2000_m3 AS DOUBLE)
                    AS built_volume_total_2000_m3,
                CAST(built_volume_total_slope_m3_per_year AS DOUBLE)
                    AS built_volume_total_slope_m3_per_year,
                LOWER(TRIM(hemisphere)) AS hemisphere,
                season,
                slot,
                CAST(beta_suhi AS DOUBLE) AS beta_suhi,
                CAST(beta_urban AS DOUBLE) AS beta_urban,
                CAST(beta_rural AS DOUBLE) AS beta_rural,
                CAST(n_valid_years AS INTEGER) AS n_valid_years
            FROM read_parquet({source_path})
            WHERE LOWER(TRIM(hemisphere)) IN ('north', 'south')
        ),
        pairs AS (
            SELECT
                summer.ID_UC_G0,
                summer.climate,
                summer.centroid_lon,
                summer.centroid_lat,
                summer.city_area_km2,
                summer.country,
                summer.GC_DEV_USR,
                summer.built_surface_2000_m2,
                summer.built_surface_slope_m2_per_year,
                summer.built_volume_total_2000_m3,
                summer.built_volume_total_slope_m3_per_year,
                summer.hemisphere,
                summer.slot,
                CASE
                    WHEN summer.hemisphere = 'north' THEN 'JJA'
                    ELSE 'DJF'
                END AS summer_season,
                CASE
                    WHEN summer.hemisphere = 'north' THEN 'DJF'
                    ELSE 'JJA'
                END AS winter_season,
                summer.n_valid_years AS n_valid_years_summer,
                winter.n_valid_years AS n_valid_years_winter,
                summer.beta_suhi AS beta_suhi_summer,
                winter.beta_suhi AS beta_suhi_winter,
                summer.beta_urban AS beta_urban_summer,
                winter.beta_urban AS beta_urban_winter,
                summer.beta_rural AS beta_rural_summer,
                winter.beta_rural AS beta_rural_winter
            FROM trends AS summer
            JOIN trends AS winter
              ON summer.ID_UC_G0 = winter.ID_UC_G0
             AND summer.slot = winter.slot
             AND summer.hemisphere = winter.hemisphere
             AND (
                 (summer.hemisphere = 'north'
                  AND summer.season = 'JJA' AND winter.season = 'DJF')
                 OR
                 (summer.hemisphere = 'south'
                  AND summer.season = 'DJF' AND winter.season = 'JJA')
             )
        )
        SELECT
            ID_UC_G0,
            climate,
            centroid_lon,
            centroid_lat,
            city_area_km2,
            country,
            GC_DEV_USR,
            built_surface_2000_m2,
            built_surface_slope_m2_per_year,
            built_volume_total_2000_m3,
            built_volume_total_slope_m3_per_year,
            hemisphere,
            slot,
            summer_season,
            winter_season,
            n_valid_years_summer,
            n_valid_years_winter,
            beta_suhi_summer,
            beta_suhi_winter,
            beta_suhi_summer - beta_suhi_winter AS gamma_suhi,
            beta_urban_summer,
            beta_urban_winter,
            beta_urban_summer - beta_urban_winter AS gamma_urban,
            beta_rural_summer,
            beta_rural_winter,
            beta_rural_summer - beta_rural_winter AS gamma_rural
        FROM pairs
        ORDER BY ID_UC_G0, slot
    """


def build_city_seasonal_trend_contrasts(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = Path(config["paths"]["seasonal_trend_file"])
    if not source.is_file():
        raise FileNotFoundError(f"Seasonal trend table not found: {source}")

    output_dir = Path(config["paths"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: "
            f"{output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}_staging_",
            dir=output_dir.parent,
        )
    )
    published = False
    connection = duckdb.connect(":memory:")
    try:
        source_counts = _validate_source(
            connection, source, config["decomposition_tolerance"]
        )
        parquet_path = staging / "city_seasonal_trend_contrasts.parquet"
        query = _contrast_query(source)
        connection.execute(
            f"COPY ({query}) TO {_quote_string(parquet_path.as_posix())} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        result_counts = connection.execute(
            """
            SELECT
                COUNT(*) AS paired_city_slots,
                COUNT(DISTINCT ID_UC_G0) AS paired_cities,
                COALESCE(
                    MAX(ABS(gamma_suhi - (gamma_urban - gamma_rural))),
                    0
                ) AS max_gamma_identity_error
            FROM read_parquet(?)
            """,
            [parquet_path.as_posix()],
        ).fetchone()
        paired_city_slots, paired_cities, max_gamma_error = result_counts
        if not paired_city_slots:
            raise ValueError(
                "No north/south city-slot pairs contain both the hemisphere-"
                "specific summer and winter trend rows."
            )
        if max_gamma_error > config["decomposition_tolerance"]:
            raise ValueError(
                "Seasonal contrast decomposition failed: "
                f"max error={max_gamma_error:.12g}, "
                f"tolerance={config['decomposition_tolerance']:.12g}."
            )
        output_schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [parquet_path.as_posix()],
        ).fetchall()
        source_counts["paired_city_slots"] = int(paired_city_slots)
        source_counts["paired_cities"] = int(paired_cities)
        source_counts["excluded_incomplete_city_slots"] = (
            source_counts["north_south_city_slot_candidates"]
            - int(paired_city_slots)
        )

        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source.name,
            "source_sha256": _sha256(source),
            "config_file": config_path.name,
            "config_sha256": _sha256(config_path),
            "seasonal_direction": {
                "north": {"summer": "JJA", "winter": "DJF"},
                "south": {"summer": "DJF", "winter": "JJA"},
            },
            "equator_policy": "exclude",
            "contrast_definition": (
                "gamma_metric = beta_metric_summer - beta_metric_winter"
            ),
            "decomposition_identity": (
                "gamma_suhi = gamma_urban - gamma_rural"
            ),
            "decomposition_tolerance": config["decomposition_tolerance"],
            "max_input_decomposition_error": (
                source_counts["max_input_decomposition_error"]
            ),
            "max_gamma_decomposition_error": float(max_gamma_error),
            "counts": source_counts,
            "output_rows": int(paired_city_slots),
            "output_columns": [row[0] for row in output_schema],
            "output_sha256": _sha256(parquet_path),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(
                f"Output directory appeared during build: {output_dir}"
            )
        staging.rename(output_dir)
        published = True
        print(
            f"[DONE] Wrote {paired_city_slots:,} city/slot seasonal contrasts "
            f"for {paired_cities:,} cities to {parquet_path.name}; "
            f"excluded {source_counts['equatorial_cities']:,} equatorial cities "
            "and "
            f"{source_counts['excluded_incomplete_city_slots']:,} incomplete "
            "city/slot pairs.",
            flush=True,
        )
        return output_dir / parquet_path.name
    finally:
        connection.close()
        if not published and staging.exists():
            shutil.rmtree(staging)

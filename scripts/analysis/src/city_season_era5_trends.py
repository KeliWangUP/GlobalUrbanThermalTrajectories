"""Fit seasonal slopes on MODIS/ERA5 paired years and common-year contrasts."""
from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from src.city_season_slot_trends import (
    SLOTS,
    _load_config as load_modis_config,
    _quote_string,
    _sha256,
)


METRICS = ("urban", "rural", "suhi", "era5")
METADATA_FIELDS = (
    "centroid_lat",
    "centroid_lon",
    "city_area_km2",
    "country",
    "GC_DEV_USR",
    "koppen_geiger_code",
    "koppen_geiger_class",
    "koppen_geiger_macro_group",
    "built_surface_2000_m2",
    "built_surface_slope_m2_per_year",
    "built_volume_total_2000_m3",
    "built_volume_total_slope_m3_per_year",
)


def _settings(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise ValueError("Config must contain a paths object.")
    paths = {}
    for key in ("seasonal_suhi_file", "modis_trend_config", "output_dir"):
        value = config["paths"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a non-empty path.")
        paths[key] = (config_path.parent / Path(value).expanduser()).resolve()
    tolerance = config.get("decomposition_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (float, int))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("decomposition_tolerance must be finite and non-negative.")
    modis = load_modis_config(paths["modis_trend_config"])
    if modis["years"]["minimum_valid_years"] < 3:
        raise ValueError("At least 3 valid years are required for trend standard errors.")
    return {
        "paths": paths,
        "years": modis["years"],
        "beta_scale_years": modis["beta_scale_years"],
        "tolerance": float(tolerance),
    }


def _validate_source(
    connection: duckdb.DuckDBPyConnection, source: Path
) -> list[str]:
    if not source.is_file():
        raise FileNotFoundError(f"Seasonal ERA5-aligned SUHI input not found: {source}")
    description = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]
    ).fetchall()
    types = {row[0]: row[1] for row in description}
    required = {
        "ID_UC_G0", "season_year", "season", "centroid_lat",
        "era5_temperature_2m_c", "era5_qc_pass",
    }
    for slot in SLOTS:
        required.update({
            f"qc_pass_season_{slot}", f"urban_{slot}_mean",
            f"rural_{slot}_mean", f"suhi_{slot}_mean",
        })
    missing = required - types.keys()
    if missing:
        raise ValueError(f"Seasonal input is missing required columns: {sorted(missing)}.")
    qc_columns = ["era5_qc_pass", *(f"qc_pass_season_{slot}" for slot in SLOTS)]
    invalid_qc = [column for column in qc_columns if types[column] != "BOOLEAN"]
    if invalid_qc:
        raise TypeError(f"Season QC fields must be BOOLEAN: {invalid_qc}.")
    source_sql = _quote_string(source.as_posix())
    invalid = connection.execute(
        f"""
        SELECT COUNT(*) FROM read_parquet({source_sql})
        WHERE ID_UC_G0 IS NULL OR try_cast(ID_UC_G0 AS BIGINT) IS NULL
           OR ID_UC_G0 <> try_cast(ID_UC_G0 AS BIGINT)
           OR season_year IS NULL OR try_cast(season_year AS INTEGER) IS NULL
           OR season_year <> try_cast(season_year AS INTEGER)
           OR season IS NULL OR season NOT IN ('DJF', 'MAM', 'JJA', 'SON')
           OR centroid_lat IS NULL OR NOT isfinite(CAST(centroid_lat AS DOUBLE))
           OR abs(CAST(centroid_lat AS DOUBLE)) > 90
        """
    ).fetchone()[0]
    duplicates = connection.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT ID_UC_G0, season_year, season FROM read_parquet({source_sql})
            GROUP BY 1, 2, 3 HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if invalid or duplicates:
        raise ValueError(
            f"Seasonal input contains {invalid} invalid rows and {duplicates} duplicate keys."
        )
    metadata = [field for field in METADATA_FIELDS if field in types]
    inconsistent_checks = " OR ".join(
        f"COUNT(DISTINCT {_quote_identifier(field)}) > 1 OR "
        f"(COUNT({_quote_identifier(field)}) > 0 AND "
        f"COUNT({_quote_identifier(field)}) < COUNT(*))"
        for field in metadata
    )
    inconsistent = connection.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT ID_UC_G0 FROM read_parquet({source_sql})
            GROUP BY ID_UC_G0 HAVING {inconsistent_checks}
        )
        """
    ).fetchone()[0]
    if inconsistent:
        raise ValueError(f"Seasonal input has inconsistent metadata for {inconsistent} cities.")
    return metadata


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _prepare_observations(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    metadata: list[str],
    start: int,
    end: int,
) -> None:
    values = ", ".join(
        f"('{slot}', qc_pass_season_{slot}, urban_{slot}_mean, "
        f"rural_{slot}_mean, suhi_{slot}_mean)"
        for slot in SLOTS
    )
    metadata_sql = ", ".join(f"s.{_quote_identifier(name)}" for name in metadata)
    connection.execute(
        f"""
        CREATE TEMP TABLE paired_observations AS
        SELECT CAST(s.ID_UC_G0 AS BIGINT) AS ID_UC_G0,
               {metadata_sql},
               CASE WHEN s.centroid_lat > 0 THEN 'north'
                    WHEN s.centroid_lat < 0 THEN 'south'
                    ELSE 'equator' END AS hemisphere,
               s.season, v.slot, CAST(s.season_year AS INTEGER) AS season_year,
               CAST(v.urban AS DOUBLE) AS urban,
               CAST(v.rural AS DOUBLE) AS rural,
               CAST(v.suhi AS DOUBLE) AS suhi,
               CAST(s.era5_temperature_2m_c AS DOUBLE) AS era5
        FROM read_parquet({_quote_string(source.as_posix())}) s
        CROSS JOIN LATERAL (VALUES {values}) v(slot, qc, urban, rural, suhi)
        WHERE s.season_year BETWEEN {start} AND {end}
          AND v.qc IS TRUE AND s.era5_qc_pass IS TRUE
          AND v.urban IS NOT NULL AND isfinite(v.urban)
          AND v.rural IS NOT NULL AND isfinite(v.rural)
          AND v.suhi IS NOT NULL AND isfinite(v.suhi)
          AND s.era5_temperature_2m_c IS NOT NULL
          AND isfinite(s.era5_temperature_2m_c)
        """
    )


def _seasonal_query(metadata: list[str], minimum: int, scale: int) -> str:
    groups = ", ".join(
        ["ID_UC_G0", *(_quote_identifier(name) for name in metadata),
         "hemisphere", "season", "slot"]
    )
    slopes = ",\n".join(
        f"regr_slope({metric}, season_year) * {scale} AS beta_{metric}"
        for metric in METRICS
    )
    standard_errors = ",\n".join(
        f"""SQRT(GREATEST(
               regr_syy({metric}, season_year)
               - regr_sxy({metric}, season_year) * regr_sxy({metric}, season_year)
                 / NULLIF(regr_sxx({metric}, season_year), 0), 0)
               / (COUNT(*) - 2) / NULLIF(regr_sxx({metric}, season_year), 0))
               * {scale} AS se_{metric}"""
        for metric in METRICS
    )
    return f"""
        SELECT {groups}, COUNT(*) AS n_valid_years,
               list(season_year ORDER BY season_year) AS valid_years,
               {slopes}, {standard_errors}
        FROM paired_observations
        GROUP BY {groups} HAVING COUNT(*) >= {minimum}
        ORDER BY ID_UC_G0, season, slot
    """


def _contrast_query(metadata: list[str], minimum: int, scale: int) -> str:
    metadata_sql = ", ".join(f"w.{_quote_identifier(name)}" for name in metadata)
    groups = ", ".join(
        ["ID_UC_G0", *(_quote_identifier(name) for name in metadata),
         "hemisphere", "slot", "summer_season", "winter_season"]
    )
    paired_values = ", ".join(
        f"w.{metric} AS {metric}_summer, c.{metric} AS {metric}_winter"
        for metric in METRICS
    )
    fitted = ", ".join(
        f"regr_slope({metric}_{season}, season_year) * {scale} "
        f"AS beta_{metric}_{season}"
        for metric in METRICS for season in ("summer", "winter")
    )
    gammas = ", ".join(
        f"beta_{metric}_summer - beta_{metric}_winter AS gamma_{metric}"
        for metric in METRICS
    )
    return f"""
        WITH season_pairs AS (
            SELECT w.ID_UC_G0, {metadata_sql}, w.hemisphere, w.slot,
                   w.season AS summer_season, c.season AS winter_season,
                   w.season_year, {paired_values}
            FROM paired_observations w
            JOIN paired_observations c
              ON w.ID_UC_G0 = c.ID_UC_G0 AND w.slot = c.slot
             AND w.season_year = c.season_year AND w.hemisphere = c.hemisphere
            WHERE (w.hemisphere = 'north' AND w.season = 'JJA' AND c.season = 'DJF')
               OR (w.hemisphere = 'south' AND w.season = 'DJF' AND c.season = 'JJA')
        ),
        fitted AS (
            SELECT {groups}, COUNT(*) AS n_common_years,
                   list(season_year ORDER BY season_year) AS common_years,
                   {fitted}
            FROM season_pairs GROUP BY {groups} HAVING COUNT(*) >= {minimum}
        )
        SELECT *, {gammas}
        FROM fitted ORDER BY ID_UC_G0, slot
    """


def build_city_season_era5_trends(config_path: Path) -> dict[str, Any]:
    """Create seasonal slopes and separately refitted common-year contrasts."""
    config_path = config_path.resolve()
    config = _settings(config_path)
    paths = config["paths"]
    source = paths["seasonal_suhi_file"]
    output_dir = paths["output_dir"]
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    try:
        metadata = _validate_source(connection, source)
        years = config["years"]
        _prepare_observations(connection, source, metadata, years["start"], years["end"])
        obs_count = connection.execute("SELECT COUNT(*) FROM paired_observations").fetchone()[0]
        print(f"[INFO] MODIS/ERA5 paired seasonal observations: {obs_count:,}", flush=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}_staging_", dir=output_dir.parent
        ) as directory:
            stage = Path(directory) / "result"
            stage.mkdir()
            outputs = {
                "seasonal": stage / "city_season_era5_trends.parquet",
                "contrast": stage / "city_season_era5_contrasts.parquet",
            }
            queries = {
                "seasonal": _seasonal_query(
                    metadata, years["minimum_valid_years"], config["beta_scale_years"]
                ),
                "contrast": _contrast_query(
                    metadata, years["minimum_valid_years"], config["beta_scale_years"]
                ),
            }
            result_info: dict[str, Any] = {}
            for kind, path in outputs.items():
                connection.execute(
                    f"COPY ({queries[kind]}) TO {_quote_string(path.as_posix())} "
                    "(FORMAT PARQUET, COMPRESSION ZSTD)"
                )
                stats = connection.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT ID_UC_G0) FROM read_parquet(?)",
                    [str(path)],
                ).fetchone()
                if not stats[0]:
                    raise ValueError(f"No {kind} groups met the paired-year requirements.")
                columns = [
                    row[0] for row in connection.execute(
                        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
                    ).fetchall()
                ]
                result_info[kind] = {
                    "file": path.name, "rows": stats[0], "cities": stats[1],
                    "columns": columns, "sha256": _sha256(path),
                }
                print(f"[CHECK] {kind}: {stats[0]:,} rows, {stats[1]:,} cities", flush=True)
            errors = connection.execute(
                """
                SELECT MAX(abs(gamma_suhi - (gamma_urban - gamma_rural))),
                       MAX(abs(gamma_era5 - (beta_era5_summer - beta_era5_winter)))
                FROM read_parquet(?)
                """,
                [str(outputs["contrast"])],
            ).fetchone()
            seasonal_error = connection.execute(
                """
                SELECT MAX(abs(beta_suhi - (beta_urban - beta_rural)))
                FROM read_parquet(?)
                """,
                [str(outputs["seasonal"])],
            ).fetchone()[0]
            if any(value is None or value > config["tolerance"] for value in (*errors, seasonal_error)):
                raise ValueError(
                    f"Trend/contrast decomposition failed: seasonal={seasonal_error}, "
                    f"contrast={errors}, tolerance={config['tolerance']}."
                )
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_file": str(source),
                "source_sha256": _sha256(source),
                "config_file": str(config_path),
                "config_sha256": _sha256(config_path),
                "modis_trend_config": str(paths["modis_trend_config"]),
                "modis_trend_config_sha256": _sha256(paths["modis_trend_config"]),
                "years": years,
                "beta_scale_years": config["beta_scale_years"],
                "beta_units": f"degrees_Celsius_per_{config['beta_scale_years']}_years",
                "seasonal_year_policy": (
                    "Per city/season/slot, use MODIS QC-passing years with finite "
                    "urban/rural/SUHI and QC-passing finite ERA5 values."
                ),
                "contrast_year_policy": (
                    "Refit both seasons on their intersection of paired years per "
                    "city/slot; at least minimum_valid_years common years."
                ),
                "monthly_pairing": (
                    "MODIS and ERA5 seasonal means are independently aggregated; "
                    "pairing is by season_year, not by constituent months."
                ),
                "seasonal_direction": {
                    "north": {"summer": "JJA", "winter": "DJF"},
                    "south": {"summer": "DJF", "winter": "JJA"},
                },
                "equator_policy": "exclude from contrasts",
                "metadata_fields": metadata,
                "missing_optional_metadata_fields": [
                    name for name in METADATA_FIELDS if name not in metadata
                ],
                "paired_observations": obs_count,
                "max_beta_decomposition_error": seasonal_error,
                "max_gamma_decomposition_error": errors[0],
                "outputs": result_info,
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            if output_dir.exists():
                raise FileExistsError(f"Output directory appeared during build: {output_dir}")
            stage.rename(output_dir)
        print(f"[OUTPUT] {output_dir}", flush=True)
        return manifest
    finally:
        connection.close()

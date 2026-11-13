"""Build city/season trends and summer-minus-winter gamma by urban zone."""
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


SEASONS = ("DJF", "MAM", "JJA", "SON")
SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")
REQUIRED_COLUMNS = {
    "ID_UC_G0",
    "centroid_lat",
    "koppen_geiger_class",
    "climate_major",
    "urban_zone",
    "year",
    "month",
    "slot",
    "paired_valid",
    "urban_lst",
    "rural_lst",
    "suhi",
}


def _quote_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Zone-trend config must be a JSON object.")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Zone-trend config requires a 'paths' object.")
    for key in ("monthly_zone_gold", "gold_manifest", "output_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config 'paths.{key}' must be a non-empty path.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths[key] = path.resolve().as_posix()

    windows = config.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Config 'windows' must be a non-empty list.")
    normalized_windows = []
    for window in windows:
        if (
            not isinstance(window, list)
            or len(window) != 2
            or any(type(year) is not int or year < 1 for year in window)
            or window[1] < window[0]
        ):
            raise ValueError("Each window must be a positive [start, end] pair.")
        normalized_windows.append(window)
    if len({tuple(window) for window in normalized_windows}) != len(
        normalized_windows
    ):
        raise ValueError("Trend windows must not contain duplicates.")

    qc = config.get("qc")
    if not isinstance(qc, dict):
        raise ValueError("Config requires a 'qc' object.")
    for key in ("min_valid_year_fraction", "min_each_half_fraction"):
        value = qc.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Config 'qc.{key}' must be in [0, 1].")
    max_missing = qc.get("max_missing_span_years")
    if type(max_missing) is not int or max_missing < 0:
        raise ValueError("Config 'qc.max_missing_span_years' must be non-negative.")
    min_months = qc.get("min_valid_months_per_season")
    if type(min_months) is not int or min_months not in (1, 2, 3):
        raise ValueError(
            "Config 'qc.min_valid_months_per_season' must be 1, 2, or 3."
        )

    scale = config.get("beta_scale_years")
    if type(scale) is not int or scale < 1:
        raise ValueError("Config 'beta_scale_years' must be positive.")
    tolerance = config.get("decomposition_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError(
            "Config 'decomposition_tolerance' must be finite and non-negative."
        )
    groups = config.get("urban_groups")
    if (
        not isinstance(groups, list)
        or len(groups) != 3
        or any(not isinstance(group, str) or not group.strip() for group in groups)
        or len({group.strip() for group in groups}) != 3
        or "overall" not in {group.strip() for group in groups}
    ):
        raise ValueError(
            "Config 'urban_groups' must list distinct stable, development, "
            "and overall labels."
        )
    classification = config.get("urban_classification")
    if not isinstance(classification, dict):
        raise ValueError("Config requires an 'urban_classification' object.")
    baseline_year = classification.get("baseline_year")
    threshold = classification.get("stable_isa_threshold")
    if (
        type(baseline_year) is not int
        or baseline_year < 1
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Urban classification baseline year/ISA threshold is invalid."
        )
    if (
        classification.get("stable_label") != config["urban_groups"][0]
        or classification.get("development_label") != config["urban_groups"][1]
        or "overall" not in config["urban_groups"]
    ):
        raise ValueError(
            "urban_groups must match the configured stable and development labels "
            "and include 'overall'."
        )
    processing = config.get("processing")
    if not isinstance(processing, dict):
        raise ValueError("Config requires a 'processing' object.")
    threads = processing.get("duckdb_threads")
    if type(threads) is not int or threads < 1:
        raise ValueError("Config 'processing.duckdb_threads' must be positive.")
    memory = processing.get("duckdb_memory")
    if (
        not isinstance(memory, str)
        or not memory.strip()
        or not memory.upper().endswith(("MB", "GB"))
    ):
        raise ValueError("Config 'processing.duckdb_memory' must use MB or GB.")

    config["windows"] = normalized_windows
    config["urban_groups"] = [group.strip() for group in groups]
    config["decomposition_tolerance"] = float(tolerance)
    config["_config_path"] = str(config_path)
    return config


def _validate_source(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    groups: list[str],
) -> int:
    description = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [source.as_posix()],
    ).fetchall()
    available = {row[0] for row in description}
    missing = REQUIRED_COLUMNS - available
    if missing:
        raise ValueError(
            f"Monthly zone Gold table is missing fields: {sorted(missing)}."
        )

    row_count = connection.execute(
        "SELECT COUNT(*) FROM read_parquet(?)", [source.as_posix()]
    ).fetchone()[0]
    if row_count == 0:
        raise ValueError("Monthly zone Gold table contains no rows.")

    duplicates = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT ID_UC_G0, urban_zone, year, month, slot
            FROM read_parquet(?)
            GROUP BY ID_UC_G0, urban_zone, year, month, slot
            HAVING COUNT(*) > 1
        )
        """,
        [source.as_posix()],
    ).fetchone()[0]
    if duplicates:
        raise ValueError(
            "Monthly zone Gold table contains duplicate city/zone/month/slot keys."
        )

    invalid_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM read_parquet(?)
        WHERE ID_UC_G0 IS NULL
           OR urban_zone IS NULL
           OR year IS NULL OR year < 1
           OR month IS NULL OR month NOT BETWEEN 1 AND 12
           OR slot IS NULL OR slot NOT IN
              ('terra_day', 'terra_night', 'aqua_day', 'aqua_night')
           OR centroid_lat IS NULL
           OR NOT isfinite(CAST(centroid_lat AS DOUBLE))
           OR ABS(CAST(centroid_lat AS DOUBLE)) > 90
           OR paired_valid IS NULL
           OR (paired_valid AND (
                 urban_lst IS NULL OR NOT isfinite(CAST(urban_lst AS DOUBLE))
              OR rural_lst IS NULL OR NOT isfinite(CAST(rural_lst AS DOUBLE))
              OR suhi IS NULL OR NOT isfinite(CAST(suhi AS DOUBLE))
           ))
        """,
        [source.as_posix()],
    ).fetchone()[0]
    if invalid_rows:
        raise ValueError(
            f"Monthly zone Gold table contains {invalid_rows:,} invalid rows."
        )

    present_groups = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT urban_zone FROM read_parquet(?)",
            [source.as_posix()],
        ).fetchall()
    }
    if present_groups != set(groups):
        raise ValueError(
            "Monthly zone Gold groups differ from config: "
            f"configured={sorted(groups)}, found={sorted(present_groups)}."
        )

    inconsistent_metadata = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT ID_UC_G0
            FROM read_parquet(?)
            GROUP BY ID_UC_G0
            HAVING COUNT(DISTINCT centroid_lat) > 1
                OR COUNT(DISTINCT koppen_geiger_class) > 1
                OR COUNT(DISTINCT climate_major) > 1
        )
        """,
        [source.as_posix()],
    ).fetchone()[0]
    if inconsistent_metadata:
        raise ValueError(
            "Monthly zone Gold table has inconsistent city climate metadata."
        )

    qc_type = next(row[1] for row in description if row[0] == "paired_valid")
    if qc_type != "BOOLEAN":
        raise TypeError("paired_valid must have BOOLEAN type.")
    return int(row_count)


def _season_means_query(source: Path, start: int, end: int, min_months: int) -> str:
    path = _quote_string(source.as_posix())
    return f"""
        WITH monthly AS (
            SELECT
                CAST(ID_UC_G0 AS BIGINT) AS ID_UC_G0,
                CAST(centroid_lat AS DOUBLE) AS centroid_lat,
                CAST(koppen_geiger_class AS VARCHAR) AS climate,
                CAST(climate_major AS VARCHAR) AS climate_major,
                CAST(urban_zone AS VARCHAR) AS urban_zone,
                CAST(slot AS VARCHAR) AS slot,
                CAST(year AS INTEGER)
                    + CASE WHEN month = 12 THEN 1 ELSE 0 END AS season_year,
                CASE
                    WHEN month IN (12, 1, 2) THEN 'DJF'
                    WHEN month IN (3, 4, 5) THEN 'MAM'
                    WHEN month IN (6, 7, 8) THEN 'JJA'
                    ELSE 'SON'
                END AS season,
                CAST(paired_valid AS BOOLEAN) AS paired_valid,
                CAST(urban_lst AS DOUBLE) AS urban_lst,
                CAST(rural_lst AS DOUBLE) AS rural_lst,
                CAST(suhi AS DOUBLE) AS suhi
            FROM read_parquet({path})
            WHERE (year BETWEEN {start} AND {end})
               OR (year = {start - 1} AND month = 12)
        ),
        seasonal AS (
            SELECT
                ID_UC_G0,
                centroid_lat,
                climate,
                climate_major,
                urban_zone,
                slot,
                season_year,
                season,
                COUNT(*) FILTER (WHERE paired_valid) AS n_valid_months,
                AVG(urban_lst) FILTER (WHERE paired_valid) AS urban_mean,
                AVG(rural_lst) FILTER (WHERE paired_valid) AS rural_mean,
                AVG(suhi) FILTER (WHERE paired_valid) AS suhi_mean
            FROM monthly
            WHERE season_year BETWEEN {start} AND {end}
            GROUP BY
                ID_UC_G0, centroid_lat, climate, climate_major,
                urban_zone, slot, season_year, season
        )
        SELECT *,
               n_valid_months >= {min_months} AS season_qc_pass
        FROM seasonal
    """


def _season_trends_query(
    start: int,
    end: int,
    min_valid_year_fraction: float,
    min_each_half_fraction: float,
    max_missing_span_years: int,
    beta_scale_years: int,
) -> str:
    length = end - start + 1
    n_min = math.ceil(length * min_valid_year_fraction)
    middle = start + (length - 1) // 2
    first_half_min = math.ceil(
        (middle - start + 1) * min_each_half_fraction
    )
    second_half_min = math.ceil((end - middle) * min_each_half_fraction)
    min_observed_span = max(0, length - 1 - max_missing_span_years)
    return f"""
        WITH trend_raw AS (
            SELECT
                ID_UC_G0,
                climate,
                climate_major,
                centroid_lat,
                urban_zone,
                season,
                slot,
                COUNT(*) AS n_valid_years,
                MIN(season_year) AS first_year,
                MAX(season_year) AS last_year,
                COUNT(*) FILTER (
                    WHERE season_year <= {middle}
                ) AS first_half_n,
                COUNT(*) FILTER (
                    WHERE season_year > {middle}
                ) AS second_half_n,
                REGR_SLOPE(urban_mean, season_year) AS slope_urban,
                REGR_SLOPE(rural_mean, season_year) AS slope_rural,
                REGR_SLOPE(suhi_mean, season_year) AS slope_suhi,
                REGR_SXX(urban_mean, season_year) AS sxx_urban,
                REGR_SXX(rural_mean, season_year) AS sxx_rural,
                REGR_SXX(suhi_mean, season_year) AS sxx_suhi,
                REGR_SXY(urban_mean, season_year) AS sxy_urban,
                REGR_SXY(rural_mean, season_year) AS sxy_rural,
                REGR_SXY(suhi_mean, season_year) AS sxy_suhi,
                REGR_SYY(urban_mean, season_year) AS syy_urban,
                REGR_SYY(rural_mean, season_year) AS syy_rural,
                REGR_SYY(suhi_mean, season_year) AS syy_suhi
            FROM season_means
            WHERE season_qc_pass
              AND season_year BETWEEN {start} AND {end}
            GROUP BY
                ID_UC_G0, climate, climate_major, centroid_lat,
                urban_zone, season, slot
        ),
        trends AS (
            SELECT *,
                   (n_valid_years >= {n_min}
                    AND first_half_n >= {first_half_min}
                    AND second_half_n >= {second_half_min}
                    AND last_year - first_year >= {min_observed_span})
                       AS qc_pass
            FROM trend_raw
        )
        SELECT
            ID_UC_G0,
            climate,
            climate_major,
            centroid_lat,
            CASE
                WHEN centroid_lat > 0 THEN 'north'
                WHEN centroid_lat < 0 THEN 'south'
                ELSE 'equator'
            END AS hemisphere,
            urban_zone,
            season,
            slot,
            n_valid_years,
            first_year,
            last_year,
            first_half_n,
            second_half_n,
            qc_pass,
            CASE WHEN qc_pass THEN slope_urban * {beta_scale_years} END
                AS beta_urban,
            CASE WHEN qc_pass THEN slope_rural * {beta_scale_years} END
                AS beta_rural,
            CASE WHEN qc_pass THEN slope_suhi * {beta_scale_years} END
                AS beta_suhi,
            CASE WHEN qc_pass THEN SQRT(
                GREATEST(
                    syy_urban - sxy_urban * sxy_urban / NULLIF(sxx_urban, 0),
                    0
                ) / (n_valid_years - 2) / NULLIF(sxx_urban, 0)
            ) * {beta_scale_years} END AS se_urban,
            CASE WHEN qc_pass THEN SQRT(
                GREATEST(
                    syy_rural - sxy_rural * sxy_rural / NULLIF(sxx_rural, 0),
                    0
                ) / (n_valid_years - 2) / NULLIF(sxx_rural, 0)
            ) * {beta_scale_years} END AS se_rural,
            CASE WHEN qc_pass THEN SQRT(
                GREATEST(
                    syy_suhi - sxy_suhi * sxy_suhi / NULLIF(sxx_suhi, 0),
                    0
                ) / (n_valid_years - 2) / NULLIF(sxx_suhi, 0)
            ) * {beta_scale_years} END AS se_suhi
        FROM trends
        ORDER BY ID_UC_G0, urban_zone, season, slot
    """


def _gamma_query(source: Path) -> str:
    path = _quote_string(source.as_posix())
    return f"""
        WITH valid_trends AS (
            SELECT *
            FROM read_parquet({path})
            WHERE qc_pass AND hemisphere IN ('north', 'south')
        ),
        pairs AS (
            SELECT
                summer.ID_UC_G0,
                summer.climate,
                summer.climate_major,
                summer.centroid_lat,
                summer.hemisphere,
                summer.urban_zone,
                summer.slot,
                CASE WHEN summer.hemisphere = 'north'
                     THEN 'JJA' ELSE 'DJF' END AS summer_season,
                CASE WHEN summer.hemisphere = 'north'
                     THEN 'DJF' ELSE 'JJA' END AS winter_season,
                summer.beta_urban AS beta_urban_summer,
                winter.beta_urban AS beta_urban_winter,
                summer.beta_rural AS beta_rural_summer,
                winter.beta_rural AS beta_rural_winter,
                summer.beta_suhi AS beta_suhi_summer,
                winter.beta_suhi AS beta_suhi_winter
            FROM valid_trends AS summer
            JOIN valid_trends AS winter
              ON summer.ID_UC_G0 = winter.ID_UC_G0
             AND summer.urban_zone = winter.urban_zone
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
            climate_major,
            centroid_lat,
            hemisphere,
            urban_zone,
            slot,
            summer_season,
            winter_season,
            beta_urban_summer,
            beta_urban_winter,
            beta_urban_summer - beta_urban_winter AS gamma_urban,
            beta_rural_summer,
            beta_rural_winter,
            beta_rural_summer - beta_rural_winter AS gamma_rural,
            beta_suhi_summer,
            beta_suhi_winter,
            beta_suhi_summer - beta_suhi_winter AS gamma_suhi
        FROM pairs
        ORDER BY ID_UC_G0, urban_zone, slot
    """


def _output_record(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> dict[str, Any]:
    return {
        "file": path.name,
        "rows": int(
            connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [path.as_posix()]
            ).fetchone()[0]
        ),
        "schema": [
            {"name": row[0], "type": row[1]}
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [path.as_posix()]
            ).fetchall()
        ],
    }


def build_city_season_zone_trends(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = Path(config["paths"]["monthly_zone_gold"])
    if not source.is_file():
        raise FileNotFoundError(f"Monthly zone Gold table not found: {source}")
    gold_manifest = Path(config["paths"]["gold_manifest"])
    if not gold_manifest.is_file():
        raise FileNotFoundError(f"Gold manifest not found: {gold_manifest}")
    manifest_data = json.loads(gold_manifest.read_text(encoding="utf-8"))
    actual_classification = manifest_data.get("urban_classification")
    expected_classification = config["urban_classification"]
    if not isinstance(actual_classification, dict) or any(
        actual_classification.get(key) != expected_classification.get(key)
        for key in (
            "baseline_year",
            "stable_isa_threshold",
            "stable_label",
            "development_label",
        )
    ):
        raise ValueError(
            "Zone trend config classification does not match the source Gold "
            "manifest."
        )

    output_dir = Path(config["paths"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(
            f"Zone-trend output already exists; refusing to overwrite: "
            f"{output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}_staging_",
            dir=output_dir.parent,
        )
    )
    connection = duckdb.connect(":memory:")
    scratch_directory = tempfile.TemporaryDirectory(
        prefix=".city_season_zone_trends_",
        dir=output_dir.parent,
    )
    published = False
    try:
        connection.execute(
            f"SET threads = {config['processing']['duckdb_threads']}"
        )
        connection.execute(
            f"SET memory_limit = "
            f"{_quote_string(config['processing']['duckdb_memory'])}"
        )
        connection.execute(
            "SET temp_directory = "
            f"{_quote_string(Path(scratch_directory.name).as_posix())}"
        )
        source_rows = _validate_source(
            connection, source, config["urban_groups"]
        )
        outputs: list[Path] = []
        for start, end in config["windows"]:
            connection.execute(
                f"CREATE OR REPLACE TEMP TABLE season_means AS "
                + _season_means_query(
                    source,
                    start,
                    end,
                    config["qc"]["min_valid_months_per_season"],
                )
            )
            season_path = staging / f"city_season_zone_means_{start}_{end}.parquet"
            connection.execute(
                f"COPY (SELECT * FROM season_means) "
                f"TO {_quote_string(season_path.as_posix())} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )

            trend_query = _season_trends_query(
                start,
                end,
                float(config["qc"]["min_valid_year_fraction"]),
                float(config["qc"]["min_each_half_fraction"]),
                config["qc"]["max_missing_span_years"],
                config["beta_scale_years"],
            )
            connection.execute(
                f"CREATE OR REPLACE TEMP TABLE season_trends AS {trend_query}"
            )
            trend_path = staging / f"city_season_zone_trends_{start}_{end}.parquet"
            connection.execute(
                "COPY (SELECT * FROM season_trends) "
                f"TO {_quote_string(trend_path.as_posix())} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            trend_count = connection.execute(
                "SELECT COUNT(*) FROM season_trends"
            ).fetchone()[0]
            if not trend_count:
                raise ValueError(
                    f"No seasonal zone trends were generated for {start}-{end}."
                )

            trend_error = connection.execute(
                """
                SELECT COALESCE(MAX(ABS(
                    beta_suhi - (beta_urban - beta_rural)
                )), 0)
                FROM season_trends
                WHERE qc_pass
                """
            ).fetchone()[0]
            if trend_error > config["decomposition_tolerance"]:
                raise ValueError(
                    "Seasonal zone trend decomposition failed: "
                    f"error={trend_error:.12g}."
                )

            gamma_query = _gamma_query(trend_path)
            gamma_path = staging / f"city_season_zone_gamma_{start}_{end}.parquet"
            connection.execute(
                f"COPY ({gamma_query}) "
                f"TO {_quote_string(gamma_path.as_posix())} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            gamma_count = connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?)",
                [gamma_path.as_posix()],
            ).fetchone()[0]
            if not gamma_count:
                raise ValueError(
                    f"No complete summer/winter zone gamma pairs for "
                    f"{start}-{end}."
                )
            gamma_error = connection.execute(
                """
                SELECT COALESCE(MAX(ABS(
                    gamma_suhi - (gamma_urban - gamma_rural)
                )), 0)
                FROM read_parquet(?)
                """,
                [gamma_path.as_posix()],
            ).fetchone()[0]
            if gamma_error > config["decomposition_tolerance"]:
                raise ValueError(
                    "Seasonal zone gamma decomposition failed: "
                    f"error={gamma_error:.12g}."
                )
            outputs.extend((season_path, trend_path, gamma_path))

        manifest_path = staging / "manifest.json"
        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source.name,
            "source_sha256": _sha256(source),
            "gold_manifest": gold_manifest.name,
            "gold_manifest_sha256": _sha256(gold_manifest),
            "config_file": config_path.name,
            "config_sha256": _sha256(config_path),
            "source_rows": source_rows,
            "urban_groups": config["urban_groups"],
            "urban_classification": expected_classification,
            "seasons": list(SEASONS),
            "season_year_convention": (
                "December is assigned to the following year's DJF season."
            ),
            "season_aggregation": {
                "method": "equal-weight mean of valid monthly city means",
                "minimum_valid_months": config["qc"][
                    "min_valid_months_per_season"
                ],
            },
            "trend_scale": f"degrees Celsius per {config['beta_scale_years']} years",
            "summer_minus_winter": {
                "north": "JJA - DJF",
                "south": "DJF - JJA",
                "equator": "excluded",
            },
            "decomposition_tolerance": config["decomposition_tolerance"],
            "outputs": [
                _output_record(connection, path) for path in outputs
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(
                f"Zone-trend output appeared during build: {output_dir}"
            )
        staging.rename(output_dir)
        published = True
    finally:
        connection.close()
        scratch_directory.cleanup()
        if not published and staging.exists():
            shutil.rmtree(staging)
    return output_dir

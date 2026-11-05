#!/usr/bin/env python3
"""Build auditable city-month LST/SUHI trend tables from the Silver datasets."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError as exc:
    raise SystemExit("Install project dependencies first: pip install -r requirements.txt") from exc


SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")
PATH_KEYS = (
    "pixel_lst",
    "pixel_development",
    "rural_lst",
    "city_metadata",
    "priority_city_ids",
    "silver_manifest",
    "output_dir",
)


def ident(name: str) -> str:
    """Quote a SQL identifier from the config."""
    return '"' + name.replace('"', '""') + '"'


def literal(value: str) -> str:
    """Quote a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def parquet_scan(path: str) -> str:
    """Create a Parquet scan that supports city Hive partitions and schema unions."""
    return (
        f"read_parquet({literal(path)}, "
        "union_by_name=true, hive_partitioning=true)"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def load_config(config_path: Path) -> dict[str, Any]:
    """Load JSON and resolve configured paths relative to that JSON file."""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Gold config root must be a JSON object.")

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Gold config 'paths' must be an object.")
    config["_configured_paths"] = dict(paths)
    for key in PATH_KEYS:
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Gold config 'paths.{key}' must be a non-empty path.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths[key] = path.resolve().as_posix()

    config["_config_path"] = str(config_path)
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    study = config.get("study")
    processing = config.get("processing")
    qc = config.get("qc")
    windows = config.get("windows")
    if not isinstance(study, dict) or not isinstance(processing, dict):
        raise ValueError("Gold config requires 'study' and 'processing' objects.")
    if not isinstance(qc, dict) or not isinstance(windows, list) or not windows:
        raise ValueError("Gold config requires a non-empty 'windows' list and 'qc' object.")

    start = _positive_integer(study.get("start_year"), "study.start_year")
    end = _positive_integer(study.get("end_year"), "study.end_year")
    if end < start:
        raise ValueError("study.end_year must be at least study.start_year.")
    _positive_integer(study.get("expected_city_count"), "study.expected_city_count")
    _positive_integer(processing.get("duckdb_threads"), "processing.duckdb_threads")
    memory = processing.get("duckdb_memory")
    if not isinstance(memory, str) or re.fullmatch(
        r"[1-9]\d*(?:MB|GB)", memory.upper()
    ) is None:
        raise ValueError("processing.duckdb_memory must look like '1GB' or '512MB'.")

    numeric_ranges = {
        "max_water_fraction": (0.0, 1.0),
        "min_urban_pixel_coverage": (0.0, 1.0),
        "min_valid_year_fraction": (0.0, 1.0),
        "min_each_half_fraction": (0.0, 1.0),
    }
    for name, (minimum, maximum) in numeric_ranges.items():
        value = qc.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"qc.{name} must be numeric.")
        if not minimum <= float(value) <= maximum:
            raise ValueError(f"qc.{name} must be in [{minimum}, {maximum}].")

    for name in (
        "min_pixel_clear_days",
        "min_pixel_n8day",
        "min_rural_clear_days",
        "min_rural_n8day",
    ):
        value = qc.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"qc.{name} must be a non-negative number.")
    for name in ("min_urban_pixels", "min_rural_pixels"):
        value = qc.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"qc.{name} must be a positive integer.")
    missing_span = qc.get("max_missing_span_years")
    if type(missing_span) is not int or missing_span < 0:
        raise ValueError("qc.max_missing_span_years must be a non-negative integer.")

    for window in windows:
        if (
            not isinstance(window, list)
            or len(window) != 2
            or any(type(year) is not int for year in window)
        ):
            raise ValueError("Each trend window must be a two-integer [start, end] list.")
        window_start, window_end = window
        if window_start < start or window_end > end or window_end < window_start:
            raise ValueError(f"Trend window {window} is outside the configured study years.")
        if missing_span >= window_end - window_start + 1:
            raise ValueError(
                f"qc.max_missing_span_years is too large for trend window {window}."
            )

    for section in ("columns", "slots"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Gold config '{section}' must be an object.")
    for section in ("pixel", "development", "rural", "climate"):
        if not isinstance(config["columns"].get(section), dict):
            raise ValueError(f"Gold config 'columns.{section}' must be an object.")
    for section in ("pixel", "rural"):
        if not isinstance(config["slots"].get(section), dict):
            raise ValueError(f"Gold config 'slots.{section}' must be an object.")
        missing_slots = set(SLOTS) - set(config["slots"][section])
        if missing_slots:
            raise ValueError(
                f"Gold config 'slots.{section}' is missing {sorted(missing_slots)}."
            )
        required_fields = (
            {"lst", "clear", "n8", "valid_pixels"}
            if section == "rural"
            else {"lst", "clear", "n8"}
        )
        for slot in SLOTS:
            mapping = config["slots"][section][slot]
            if (
                not isinstance(mapping, dict)
                or not required_fields.issubset(mapping)
                or any(not isinstance(value, str) or not value for value in mapping.values())
            ):
                raise ValueError(
                    f"Gold config 'slots.{section}.{slot}' must map "
                    f"{sorted(required_fields)} to non-empty field names."
                )
    for section, mapping in config["columns"].items():
        if any(not isinstance(value, str) or not value for value in mapping.values()):
            raise ValueError(
                f"Gold config 'columns.{section}' values must be non-empty field names."
            )


def check_columns(
    connection: duckdb.DuckDBPyConnection,
    scan_sql: str,
    mapping: dict[str, str],
    label: str,
) -> None:
    actual = {
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM {scan_sql}"
        ).fetchall()
    }
    missing = sorted(set(mapping.values()) - actual)
    if missing:
        raise ValueError(
            f"{label} lacks configured fields {missing}; available fields: "
            f"{sorted(actual)}."
        )
    print(f"[SCHEMA OK] {label}: {len(actual)} fields.", flush=True)


def load_priority_city_ids(config: dict[str, Any]) -> list[int]:
    path = Path(config["paths"]["priority_city_ids"])
    values = json.loads(path.read_text(encoding="utf-8"))
    expected_count = config["study"]["expected_city_count"]
    if (
        not isinstance(values, list)
        or len(values) != expected_count
        or any(type(city_id) is not int or city_id < 0 for city_id in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "Priority city JSON must contain exactly "
            f"{expected_count} unique non-negative integer IDs."
        )
    return sorted(values)


def validate_inputs(config: dict[str, Any]) -> list[int]:
    """Fail before creating outputs if a configured input is missing."""
    paths = config["paths"]
    for key in ("pixel_development", "city_metadata", "priority_city_ids", "silver_manifest"):
        if not Path(paths[key]).is_file():
            raise FileNotFoundError(f"Gold input file not found: {paths[key]}")

    for key in ("pixel_lst", "rural_lst"):
        files = [Path(file) for file in glob.glob(paths[key], recursive=True)]
        if not files:
            raise FileNotFoundError(f"Gold input pattern matched no files: {paths[key]}")
        if key == "pixel_lst" and len(files) != config["study"]["expected_city_count"]:
            raise ValueError(
                f"Pixel LST pattern matched {len(files)} city files; expected "
                f"{config['study']['expected_city_count']}."
            )
        if key == "rural_lst":
            expected_years = (
                config["study"]["end_year"] - config["study"]["start_year"] + 1
            )
            if len(files) != expected_years:
                raise ValueError(
                    f"Rural LST pattern matched {len(files)} files; expected "
                    f"{expected_years} annual files."
                )
    return load_priority_city_ids(config)


def create_city_pixels(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    priority_city_ids: list[int],
) -> None:
    columns = config["columns"]["development"]
    source = parquet_scan(config["paths"]["pixel_development"])
    check_columns(connection, source, columns, "pixel_development")
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE development_pixels AS
        SELECT CAST({ident(columns['city'])} AS BIGINT) AS city_id,
               CAST({ident(columns['pixel_id'])} AS VARCHAR) AS pixel_id,
               CAST({ident(columns['water_fraction'])} AS DOUBLE) AS water_fraction
        FROM {source}
    """)

    row_count, unique_count, bad_keys, bad_water = connection.execute("""
        SELECT COUNT(*),
               COUNT(DISTINCT (city_id, pixel_id)),
               COUNT(*) FILTER (
                   WHERE city_id IS NULL OR pixel_id IS NULL
                      OR NOT regexp_full_match(
                          pixel_id, 'MODIS_-?[0-9]+_-?[0-9]+'
                      )
               ),
               COUNT(*) FILTER (
                  WHERE water_fraction IS NOT NULL
                    AND (NOT isfinite(water_fraction)
                         OR water_fraction < 0 OR water_fraction > 1)
               )
        FROM development_pixels
    """).fetchone()
    if row_count == 0 or row_count != unique_count or bad_keys:
        raise ValueError(
            "Development pixels have empty, duplicate, null, or invalid keys/"
            f"water fractions: rows={row_count}, unique={unique_count}, "
            f"bad_keys={bad_keys}."
        )
    if bad_water:
        raise ValueError(
            f"Development data has {bad_water} non-finite or out-of-range "
            "water fractions."
        )

    city_ids = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT city_id FROM development_pixels"
        ).fetchall()
    }
    if city_ids != set(priority_city_ids):
        raise ValueError(
            "Development city IDs differ from the priority list; "
            f"missing={sorted(set(priority_city_ids) - city_ids)[:20]}, "
            f"unexpected={sorted(city_ids - set(priority_city_ids))[:20]}."
        )

    threshold = float(config["qc"]["max_water_fraction"])
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE eligible_pixels AS
        SELECT city_id, pixel_id
        FROM development_pixels
        WHERE water_fraction IS NOT NULL
          AND water_fraction <= {threshold}
    """)
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE city_pixel_stats AS
        SELECT d.city_id,
               COUNT(*) AS n_pixels_total,
               COUNT(e.pixel_id) AS n_pixels_eligible,
               AVG(d.water_fraction) AS mean_water_fraction,
               COUNT(*) FILTER (WHERE d.water_fraction IS NULL)
                   AS n_pixels_missing_water
        FROM development_pixels d
        LEFT JOIN eligible_pixels e
          ON d.city_id = e.city_id
         AND d.pixel_id = e.pixel_id
        GROUP BY d.city_id
    """)
    no_eligible = connection.execute("""
        SELECT city_id FROM city_pixel_stats
        WHERE n_pixels_eligible = 0
        ORDER BY city_id LIMIT 10
    """).fetchall()
    if no_eligible:
        raise ValueError(f"Cities with no water-eligible pixels: {no_eligible}.")

    stats = connection.execute("""
        SELECT COUNT(*), SUM(n_pixels_total), SUM(n_pixels_eligible),
               SUM(n_pixels_missing_water)
        FROM city_pixel_stats
    """).fetchone()
    if stats[0] != config["study"]["expected_city_count"]:
        raise ValueError(
            f"Expected {config['study']['expected_city_count']} cities in "
            f"development, found {stats[0]}."
        )
    print(
        f"[PIXEL MASK] cities={stats[0]:,}; total={stats[1]:,}; "
        f"eligible={stats[2]:,}; missing water={stats[3]:,}",
        flush=True,
    )


def pixel_slot_aggregates(config: dict[str, Any], slot: str) -> str:
    mapping = config["slots"]["pixel"][slot]
    valid = (
        f"p.{ident(mapping['lst'])} IS NOT NULL AND "
        f"p.{ident(mapping['clear'])} >= "
        f"{float(config['qc']['min_pixel_clear_days'])} AND "
        f"p.{ident(mapping['n8'])} >= "
        f"{float(config['qc']['min_pixel_n8day'])}"
    )
    return (
        f"AVG(p.{ident(mapping['lst'])}) FILTER (WHERE {valid}) "
        f"AS {ident(slot + '_lst')}, "
        f"COUNT(*) FILTER (WHERE {valid}) AS {ident(slot + '_valid_n')}"
    )


def create_urban_month(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
) -> None:
    source = parquet_scan(config["paths"]["pixel_lst"])
    columns = config["columns"]["pixel"]
    required = dict(columns)
    for slot, names in config["slots"]["pixel"].items():
        required.update({f"{slot}.{name}": field for name, field in names.items()})
    check_columns(connection, source, required, "pixel_lst")

    slot_sql = ",\n               ".join(
        pixel_slot_aggregates(config, slot) for slot in SLOTS
    )
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE urban_month AS
        SELECT CAST(p.{ident(columns['city'])} AS BIGINT) AS city_id,
               CAST(p.{ident(columns['year'])} AS INTEGER) AS year,
               CAST(p.{ident(columns['month'])} AS INTEGER) AS month,
               COUNT(*) AS input_pixel_rows,
               {slot_sql}
        FROM {source} AS p
        JOIN eligible_pixels AS e
          ON CAST(p.{ident(columns['city'])} AS BIGINT) = e.city_id
         AND CAST(p.{ident(columns['pixel_id'])} AS VARCHAR) = e.pixel_id
        GROUP BY 1, 2, 3
    """)

    mismatches = connection.execute("""
        SELECT COUNT(*)
        FROM urban_month u
        JOIN city_pixel_stats s USING (city_id)
        WHERE u.input_pixel_rows != s.n_pixels_eligible
    """).fetchone()[0]
    if mismatches:
        raise ValueError(
            f"{mismatches} city-months do not contain exactly the eligible "
            "development pixel count."
        )

    start = config["study"]["start_year"]
    end = config["study"]["end_year"]
    expected_rows = (
        config["study"]["expected_city_count"] * (end - start + 1) * 12
    )
    actual_rows = connection.execute(
        "SELECT COUNT(*) FROM urban_month"
    ).fetchone()[0]
    invalid_dates = connection.execute(f"""
        SELECT COUNT(*) FROM urban_month
        WHERE year < {start} OR year > {end} OR month < 1 OR month > 12
    """).fetchone()[0]
    if actual_rows != expected_rows or invalid_dates:
        raise ValueError(
            f"Urban city-month coverage is incomplete or out of range: "
            f"rows={actual_rows}, expected={expected_rows}, "
            f"invalid_dates={invalid_dates}."
        )
    print(f"[AGGREGATE] {actual_rows:,} urban city-month rows.", flush=True)


def _rural_select(config: dict[str, Any]) -> str:
    columns = config["columns"]["rural"]
    expressions = [
        f"CAST({ident(columns['city'])} AS BIGINT) AS city_id",
        f"CAST({ident(columns['year'])} AS INTEGER) AS year",
        f"CAST({ident(columns['month'])} AS INTEGER) AS month",
    ]
    for slot in SLOTS:
        mapping = config["slots"]["rural"][slot]
        expressions.extend(
            (
                f"CAST({ident(mapping['lst'])} AS DOUBLE) AS {ident(slot + '_lst')}",
                f"CAST({ident(mapping['clear'])} AS DOUBLE) AS {ident(slot + '_clear')}",
                f"CAST({ident(mapping['n8'])} AS DOUBLE) AS {ident(slot + '_n8')}",
                f"CAST({ident(mapping['valid_pixels'])} AS BIGINT) "
                f"AS {ident(slot + '_valid_n')}",
            )
        )
    return ",\n               ".join(expressions)


def create_paired_months(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    priority_city_ids: list[int],
) -> None:
    rural_source = parquet_scan(config["paths"]["rural_lst"])
    rural_columns = dict(config["columns"]["rural"])
    for slot, names in config["slots"]["rural"].items():
        rural_columns.update({f"{slot}.{name}": field for name, field in names.items()})
    check_columns(connection, rural_source, rural_columns, "rural_lst")

    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE rural_month AS
        SELECT {_rural_select(config)}
        FROM {rural_source}
    """)
    duplicate_rows = connection.execute("""
        SELECT city_id, year, month, COUNT(*)
        FROM rural_month
        GROUP BY 1, 2, 3
        HAVING COUNT(*) > 1
        LIMIT 3
    """).fetchall()
    if duplicate_rows:
        raise ValueError(f"Duplicate rural city/year/month keys: {duplicate_rows}.")

    rural_city_ids = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT city_id FROM rural_month"
        ).fetchall()
    }
    if rural_city_ids != set(priority_city_ids):
        raise ValueError(
            "Rural city IDs differ from the priority list; "
            f"missing={sorted(set(priority_city_ids) - rural_city_ids)[:20]}, "
            f"unexpected={sorted(rural_city_ids - set(priority_city_ids))[:20]}."
        )

    start = config["study"]["start_year"]
    end = config["study"]["end_year"]
    expected_rows = (
        config["study"]["expected_city_count"] * (end - start + 1) * 12
    )
    rural_rows = connection.execute("SELECT COUNT(*) FROM rural_month").fetchone()[0]
    if rural_rows != expected_rows:
        raise ValueError(
            f"Rural data has {rural_rows:,} rows; expected {expected_rows:,}."
        )

    missing_urban = connection.execute("""
        SELECT COUNT(*) FROM (
            SELECT city_id, year, month FROM urban_month
            EXCEPT
            SELECT city_id, year, month FROM rural_month
        )
    """).fetchone()[0]
    missing_rural = connection.execute("""
        SELECT COUNT(*) FROM (
            SELECT city_id, year, month FROM rural_month
            EXCEPT
            SELECT city_id, year, month FROM urban_month
        )
    """).fetchone()[0]
    if missing_urban or missing_rural:
        raise ValueError(
            f"Urban/rural city-month keys differ: "
            f"urban_without_rural={missing_urban}, "
            f"rural_without_urban={missing_rural}."
        )

    qc = config["qc"]
    parts = []
    for slot in SLOTS:
        urban_valid = (
            f"u.{ident(slot + '_valid_n')} >= {int(qc['min_urban_pixels'])} "
            f"AND u.{ident(slot + '_valid_n')} * 1.0 / "
            f"NULLIF(s.n_pixels_eligible, 0) >= "
            f"{float(qc['min_urban_pixel_coverage'])} "
            f"AND u.{ident(slot + '_lst')} IS NOT NULL"
        )
        rural_valid = (
            f"r.{ident(slot + '_lst')} IS NOT NULL "
            f"AND r.{ident(slot + '_clear')} >= "
            f"{float(qc['min_rural_clear_days'])} "
            f"AND r.{ident(slot + '_n8')} >= "
            f"{float(qc['min_rural_n8day'])} "
            f"AND r.{ident(slot + '_valid_n')} >= "
            f"{int(qc['min_rural_pixels'])}"
        )
        parts.append(f"""
            SELECT u.city_id, u.year, u.month, {literal(slot)} AS slot,
                   u.{ident(slot + '_valid_n')} AS urban_valid_n,
                   s.n_pixels_eligible AS urban_eligible_n,
                   u.{ident(slot + '_valid_n')} * 1.0 /
                       NULLIF(s.n_pixels_eligible, 0) AS urban_coverage,
                   r.{ident(slot + '_valid_n')} AS rural_valid_n,
                   COALESCE(({urban_valid}), FALSE) AS urban_qc_pass,
                   COALESCE(({rural_valid}), FALSE) AS rural_qc_pass,
                   CASE WHEN {urban_valid}
                        THEN u.{ident(slot + '_lst')} END AS urban_lst,
                   CASE WHEN {rural_valid}
                        THEN r.{ident(slot + '_lst')} END AS rural_lst
            FROM urban_month u
            JOIN rural_month r
              ON u.city_id = r.city_id
             AND u.year = r.year
             AND u.month = r.month
            JOIN city_pixel_stats s ON s.city_id = u.city_id
        """)
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE paired_months AS "
        "SELECT *, urban_lst - rural_lst AS suhi, "
        "urban_qc_pass AND rural_qc_pass AS paired_valid "
        "FROM (" + " UNION ALL ".join(parts) + ") paired"
    )

    paired_rows = connection.execute(
        "SELECT COUNT(*) FROM paired_months"
    ).fetchone()[0]
    expected_paired_rows = expected_rows * len(SLOTS)
    if paired_rows != expected_paired_rows:
        raise ValueError(
            f"Paired data has {paired_rows:,} rows; "
            f"expected {expected_paired_rows:,}."
        )
    print(f"[PAIR] {paired_rows:,} city-month-slot rows.", flush=True)


def create_climate_info(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    priority_city_ids: list[int],
) -> None:
    source = parquet_scan(config["paths"]["city_metadata"])
    columns = config["columns"]["climate"]
    check_columns(connection, source, columns, "city_metadata")
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE climate_info AS
        SELECT CAST({ident(columns['city'])} AS BIGINT) AS city_id,
               CAST({ident(columns['centroid_lat'])} AS DOUBLE) AS centroid_lat,
               CAST({ident(columns['koppen_code'])} AS INTEGER)
                   AS koppen_geiger_code,
               CAST({ident(columns['koppen_class'])} AS VARCHAR)
                   AS koppen_geiger_class,
               CAST({ident(columns['macro_group'])} AS VARCHAR)
                   AS koppen_geiger_macro_group
        FROM {source}
        WHERE CAST({ident(columns['city'])} AS BIGINT)
              IN (SELECT unnest(?))
    """, [priority_city_ids])

    duplicates = connection.execute("""
        SELECT city_id, COUNT(*)
        FROM climate_info
        GROUP BY city_id
        HAVING COUNT(*) > 1
        LIMIT 3
    """).fetchall()
    if duplicates:
        raise ValueError(f"City climate metadata has duplicate city IDs: {duplicates}.")

    climate_city_ids = {
        row[0]
        for row in connection.execute(
            "SELECT city_id FROM climate_info"
        ).fetchall()
    }
    if climate_city_ids != set(priority_city_ids):
        raise ValueError(
            "Climate metadata does not cover the priority city list; "
            f"missing={sorted(set(priority_city_ids) - climate_city_ids)[:20]}."
        )

    invalid = connection.execute("""
        SELECT COUNT(*) FROM climate_info
        WHERE centroid_lat IS NULL OR centroid_lat < -90 OR centroid_lat > 90
           OR (koppen_geiger_code IS NOT NULL
               AND koppen_geiger_code NOT BETWEEN 1 AND 30)
           OR ((koppen_geiger_code IS NULL)
               != (koppen_geiger_class IS NULL))
           OR ((koppen_geiger_code IS NULL)
               != (koppen_geiger_macro_group IS NULL))
    """).fetchone()[0]
    if invalid:
        raise ValueError(
            f"City climate metadata has {invalid} invalid coordinate/class rows."
        )


def create_trends(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    start: int,
    end: int,
) -> None:
    qc = config["qc"]
    length = end - start + 1
    n_min = math.ceil(length * float(qc["min_valid_year_fraction"]))
    middle = start + (length - 1) // 2
    first_half_min = math.ceil(
        (middle - start + 1) * float(qc["min_each_half_fraction"])
    )
    second_half_min = math.ceil(
        (end - middle) * float(qc["min_each_half_fraction"])
    )
    min_observed_span = length - 1 - int(qc["max_missing_span_years"])
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE trend_raw AS
        SELECT city_id, month, slot,
               COUNT(*) AS n_years,
               MIN(year) AS first_year,
               MAX(year) AS last_year,
               COUNT(*) FILTER (WHERE year <= {middle}) AS first_half_n,
               COUNT(*) FILTER (WHERE year > {middle}) AS second_half_n,
               REGR_SLOPE(urban_lst, year) * 10.0 AS urban_trend_decade,
               REGR_SLOPE(rural_lst, year) * 10.0 AS rural_trend_decade,
               REGR_SLOPE(suhi, year) * 10.0 AS suhi_trend_decade,
               REGR_R2(suhi, year) AS suhi_trend_r2,
               AVG(urban_coverage) AS mean_urban_coverage
        FROM paired_months
        WHERE year BETWEEN {start} AND {end}
          AND paired_valid
        GROUP BY city_id, month, slot
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE trends AS
        SELECT *,
          (n_years >= {n_min}
           AND first_half_n >= {first_half_min}
           AND second_half_n >= {second_half_min}
           AND last_year - first_year >= {min_observed_span}) AS qc_pass
        FROM trend_raw
    """)
    total, passed = connection.execute("""
        SELECT COUNT(*), COUNT(*) FILTER (WHERE qc_pass)
        FROM trends
    """).fetchone()
    print(
        f"[TREND {start}-{end}] valid years>={n_min}; "
        f"half-years>={first_half_min}/{second_half_min}; "
        f"observed span>={min_observed_span}; "
        f"{passed:,}/{total:,} fits pass QC.",
        flush=True,
    )


def write_trend_outputs(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    output_dir: Path,
    start: int,
    end: int,
) -> list[Path]:
    label = f"{start}_{end}"
    climate_columns = """
        c.centroid_lat, c.koppen_geiger_code, c.koppen_geiger_class,
        c.koppen_geiger_macro_group,
        LEFT(c.koppen_geiger_class, 1) AS climate_major
    """
    long_path = output_dir / f"city_month_slot_trends_{label}.parquet"
    connection.execute(f"""
        COPY (
          SELECT t.city_id AS ID_UC_G0,
                 {climate_columns},
                 t.month, t.slot, t.n_years, t.first_year, t.last_year,
                 t.first_half_n, t.second_half_n, t.mean_urban_coverage,
                 t.qc_pass,
                 CASE WHEN t.qc_pass THEN t.urban_trend_decade END
                     AS urban_trend_decade,
                 CASE WHEN t.qc_pass THEN t.rural_trend_decade END
                     AS rural_trend_decade,
                 CASE WHEN t.qc_pass THEN t.suhi_trend_decade END
                     AS suhi_trend_decade,
                 CASE WHEN t.qc_pass THEN t.suhi_trend_r2 END
                     AS suhi_trend_r2
          FROM trends t
          JOIN climate_info c USING (city_id)
        ) TO {literal(long_path.as_posix())}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    expressions = []
    slot_codes = {
        "terra_day": "td",
        "terra_night": "tn",
        "aqua_day": "ad",
        "aqua_night": "an",
    }
    for month in range(1, 13):
        for slot in SLOTS:
            code = slot_codes[slot]
            match = f"t.month = {month} AND t.slot = {literal(slot)}"
            for measure in ("urban", "rural", "suhi"):
                name = f"{measure}_{code}_m{month:02d}_trend_decade"
                expressions.append(
                    f"MAX(CASE WHEN {match} AND t.qc_pass "
                    f"THEN t.{measure}_trend_decade END) AS {ident(name)}"
                )
            years_name = f"n_years_{code}_m{month:02d}"
            expressions.append(
                f"COALESCE(MAX(CASE WHEN {match} THEN t.n_years END), 0) "
                f"AS {ident(years_name)}"
            )
            qc_name = f"qc_{code}_m{month:02d}"
            expressions.append(
                f"COALESCE(BOOL_OR({match} AND t.qc_pass), FALSE) "
                f"AS {ident(qc_name)}"
            )

    wide_path = output_dir / f"city_month_trend_wide_{label}.parquet"
    connection.execute(f"""
        COPY (
          SELECT s.city_id AS ID_UC_G0,
                 {climate_columns},
                 s.n_pixels_total,
                 s.n_pixels_eligible,
                 s.n_pixels_missing_water,
                 s.mean_water_fraction,
                 1.0 - s.n_pixels_eligible * 1.0 /
                     NULLIF(s.n_pixels_total, 0) AS pixel_excluded_fraction,
                 {", ".join(expressions)}
          FROM city_pixel_stats s
          JOIN climate_info c USING (city_id)
          LEFT JOIN trends t USING (city_id)
          GROUP BY s.city_id, c.centroid_lat, c.koppen_geiger_code,
                   c.koppen_geiger_class, c.koppen_geiger_macro_group,
                   s.n_pixels_total, s.n_pixels_eligible,
                   s.n_pixels_missing_water, s.mean_water_fraction
        ) TO {literal(wide_path.as_posix())}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    city_count, missing_climate = connection.execute(f"""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE koppen_geiger_class IS NULL)
        FROM read_parquet({literal(wide_path.as_posix())})
    """).fetchone()
    expected = config["study"]["expected_city_count"]
    if city_count != expected:
        raise ValueError(f"Gold wide table has {city_count} cities; expected {expected}.")
    print(
        f"[OUTPUT] {wide_path}: {city_count:,} cities; "
        f"climate raster no-data={missing_climate:,}.",
        flush=True,
    )
    print(f"[OUTPUT] {long_path}: trend diagnostics and slopes.", flush=True)
    return [long_path, wide_path]


def _output_record(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> dict[str, Any]:
    count = connection.execute(
        f"SELECT COUNT(*) FROM read_parquet({literal(path.as_posix())})"
    ).fetchone()[0]
    schema = [
        {"name": row[0], "type": row[1]}
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({literal(path.as_posix())})"
        ).fetchall()
    ]
    return {
        "file": path.name,
        "rows": count,
        "bytes": path.stat().st_size,
        "schema": schema,
    }


def build_gold(config_path: Path) -> Path:
    """Build outputs in a staging directory and publish only a complete Gold set."""
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(
            f"Gold destination already exists; refusing to overwrite: {output_dir}"
        )
    priority_city_ids = validate_inputs(config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}_staging_",
            dir=output_dir.parent,
        )
    )
    print(f"[INFO] Gold staging directory: {staging}", flush=True)

    connection = duckdb.connect(":memory:")
    scratch_directory = tempfile.TemporaryDirectory(
        prefix=".city_month_gold_duckdb_",
        dir=output_dir.parent,
    )
    try:
        connection.execute(
            f"SET threads = {config['processing']['duckdb_threads']}"
        )
        connection.execute(
            f"SET memory_limit = {literal(config['processing']['duckdb_memory'])}"
        )
        connection.execute(
            f"SET temp_directory = "
            f"{literal(Path(scratch_directory.name).as_posix())}"
        )

        create_city_pixels(connection, config, priority_city_ids)
        create_urban_month(connection, config)
        create_paired_months(connection, config, priority_city_ids)
        create_climate_info(connection, config, priority_city_ids)

        paired_path = staging / "city_month_paired_qc.parquet"
        connection.execute(f"""
            COPY (SELECT * FROM paired_months)
            TO {literal(paired_path.as_posix())}
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print(f"[OUTPUT] {paired_path}: paired monthly QC records.", flush=True)

        output_paths = [paired_path]
        for start, end in config["windows"]:
            create_trends(connection, config, start, end)
            output_paths.extend(
                write_trend_outputs(connection, config, staging, start, end)
            )

        manifest_path = staging / "manifest.json"
        config_file = Path(config["_config_path"])
        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "city_count": config["study"]["expected_city_count"],
            "study_years": {
                "start": config["study"]["start_year"],
                "end": config["study"]["end_year"],
            },
            "trend_windows": config["windows"],
            "qc": config["qc"],
            "inputs": {
                "configured_paths": config["_configured_paths"],
                "config_sha256": sha256_file(config_file),
                "priority_city_ids_sha256": sha256_file(
                    Path(config["paths"]["priority_city_ids"])
                ),
                "silver_manifest_sha256": sha256_file(
                    Path(config["paths"]["silver_manifest"])
                ),
                "city_metadata_sha256": sha256_file(
                    Path(config["paths"]["city_metadata"])
                ),
            },
            "outputs": [
                _output_record(connection, path) for path in output_paths
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        try:
            connection.close()
        finally:
            scratch_directory.cleanup()

    if output_dir.exists():
        raise FileExistsError(
            f"Gold destination appeared during the build: {output_dir}"
        )
    staging.rename(output_dir)
    print(f"[DONE] Gold dataset published at {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("city_month_gold_config.json"),
        help="Path to the Gold config JSON.",
    )
    args = parser.parse_args()
    build_gold(args.config)


if __name__ == "__main__":
    main()

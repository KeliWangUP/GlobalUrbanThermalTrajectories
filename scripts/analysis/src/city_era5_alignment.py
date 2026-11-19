"""Align ERA5-Land city-month records to existing SUHI panel keys."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import duckdb


KEYS = {
    "monthly": ("ID_UC_G0", "year", "month"),
    "seasonal": ("ID_UC_G0", "season_year", "season"),
    "annual": ("ID_UC_G0", "year"),
}
OUTPUT_COLUMNS = {
    "monthly": {"era5_temperature_2m_c", "era5_record_present"},
    "seasonal": {
        "era5_temperature_2m_c", "era5_n_available_months",
        "era5_n_valid_months", "era5_qc_pass",
    },
    "annual": {
        "era5_temperature_2m_c", "era5_n_available_months",
        "era5_n_valid_months", "era5_qc_pass",
    },
}


def _quoted_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _load_settings(config_path: Path) -> tuple[dict[str, Path], int, int, bool]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise ValueError("Config must contain a paths object.")
    paths = {}
    for key in (
        "era5_raw_dir", "monthly_suhi_file", "seasonal_suhi_file",
        "annual_suhi_file", "output_dir",
    ):
        value = config["paths"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a non-empty path.")
        path = Path(value).expanduser()
        paths[key] = (config_path.parent / path).resolve()
    minimum_season = config.get("minimum_valid_months_per_season")
    minimum_year = config.get("minimum_valid_months_per_year")
    require_complete = config.get("require_complete_city_months")
    if type(require_complete) is not bool:
        raise ValueError("require_complete_city_months must be a boolean.")
    for key, value, maximum in (
        ("minimum_valid_months_per_season", minimum_season, 3),
        ("minimum_valid_months_per_year", minimum_year, 12),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{key} must be an integer from 1 to {maximum}.")
    return paths, minimum_season, minimum_year, require_complete


def _load_era5(connection: duckdb.DuckDBPyConnection, directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No ERA5 CSV files found in {directory}.")
    required = {"city_id", "month", "temperature_2m_c"}
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as file:
            header = next(csv.reader(file), [])
        if len(header) != len(set(header)) or not required.issubset(header):
            raise ValueError(f"Invalid ERA5 CSV header in {path}: {header}.")

    connection.execute(
        """
        CREATE TEMP TABLE era5_raw AS
        SELECT city_id, month, temperature_2m_c
        FROM read_csv(?, header=true, all_varchar=true, union_by_name=true)
        """,
        [[str(path) for path in files]],
    )
    invalid = connection.execute(
        r"""
        SELECT COUNT(*) FROM era5_raw
        WHERE city_id IS NULL OR NOT regexp_full_match(city_id, '[0-9]+')
           OR try_cast(city_id AS BIGINT) IS NULL
           OR month IS NULL OR NOT regexp_full_match(month, '[0-9]{4}-(0[1-9]|1[0-2])')
           OR try_strptime(month, '%Y-%m') IS NULL
           OR (temperature_2m_c IS NOT NULL AND (
               try_cast(temperature_2m_c AS DOUBLE) IS NULL
               OR NOT isfinite(try_cast(temperature_2m_c AS DOUBLE))
           ))
        """
    ).fetchone()[0]
    if invalid:
        raise ValueError(f"ERA5 input has {invalid} invalid city/month/temperature rows.")

    connection.execute(
        """
        CREATE TEMP TABLE era5_monthly AS
        SELECT CAST(city_id AS BIGINT) AS city_id,
               CAST(substr(month, 1, 4) AS INTEGER) AS year,
               CAST(substr(month, 6, 2) AS INTEGER) AS month,
               CAST(temperature_2m_c AS DOUBLE) AS temperature_2m_c
        FROM era5_raw
        """
    )
    duplicates = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT city_id, year, month FROM era5_monthly
            GROUP BY city_id, year, month HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicates:
        raise ValueError(f"ERA5 input has {duplicates} duplicate city-month keys.")
    return files


def _validate_suhi(
    connection: duckdb.DuckDBPyConnection, kind: str, path: Path
) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"SUHI {kind} input not found: {path}")
    columns = {
        row[0] for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
    }
    missing = set(KEYS[kind]) - columns
    collision = OUTPUT_COLUMNS[kind] & columns
    if missing or collision:
        raise ValueError(
            f"SUHI {kind} missing key columns {sorted(missing)} or already "
            f"contains ERA5 output columns {sorted(collision)}."
        )
    keys = ", ".join(KEYS[kind])
    missing_key = " OR ".join(f"{key} IS NULL" for key in KEYS[kind])
    count, invalid = connection.execute(
        f"""
        SELECT COUNT(*), COUNT(*) FILTER (
            WHERE {missing_key}
               OR try_cast(ID_UC_G0 AS BIGINT) IS NULL
               OR ID_UC_G0 != try_cast(ID_UC_G0 AS BIGINT)
        ) FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchone()
    duplicates = connection.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT {keys} FROM read_parquet(?)
            GROUP BY {keys} HAVING COUNT(*) > 1
        )
        """,
        [str(path)],
    ).fetchone()[0]
    if invalid or duplicates:
        raise ValueError(
            f"SUHI {kind} has {invalid} invalid keys and {duplicates} duplicate keys."
        )
    return count


def _build_aggregates(
    connection: duckdb.DuckDBPyConnection,
    minimum_season: int,
    minimum_year: int,
) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE era5_seasonal AS
        WITH labeled AS (
            SELECT city_id,
                   year + CASE WHEN month = 12 THEN 1 ELSE 0 END AS season_year,
                   CASE WHEN month IN (12, 1, 2) THEN 'DJF'
                        WHEN month IN (3, 4, 5) THEN 'MAM'
                        WHEN month IN (6, 7, 8) THEN 'JJA'
                        ELSE 'SON' END AS season,
                   temperature_2m_c
            FROM era5_monthly
        )
        SELECT city_id, season_year, season,
               COUNT(*) AS available_months,
               COUNT(temperature_2m_c) AS valid_months,
               CASE WHEN COUNT(temperature_2m_c) >= {minimum_season}
                    THEN AVG(temperature_2m_c) END AS temperature_2m_c
        FROM labeled GROUP BY city_id, season_year, season
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE era5_annual AS
        SELECT city_id, year, COUNT(*) AS available_months,
               COUNT(temperature_2m_c) AS valid_months,
               CASE WHEN COUNT(temperature_2m_c) >= {minimum_year}
                    THEN AVG(temperature_2m_c) END AS temperature_2m_c
        FROM era5_monthly GROUP BY city_id, year
        """
    )


def _aligned_query(kind: str, source: Path, minimum: int) -> str:
    if kind == "monthly":
        extra = """
            e.temperature_2m_c AS era5_temperature_2m_c,
            e.city_id IS NOT NULL AS era5_record_present
        """
        condition = "s.ID_UC_G0 = e.city_id AND s.year = e.year AND s.month = e.month"
    else:
        extra = f"""
            e.temperature_2m_c AS era5_temperature_2m_c,
            COALESCE(e.available_months, 0) AS era5_n_available_months,
            COALESCE(e.valid_months, 0) AS era5_n_valid_months,
            COALESCE(e.valid_months, 0) >= {minimum} AS era5_qc_pass
        """
        if kind == "seasonal":
            condition = (
                "s.ID_UC_G0 = e.city_id AND s.season_year = e.season_year "
                "AND s.season = e.season"
            )
        else:
            condition = "s.ID_UC_G0 = e.city_id AND s.year = e.year"
    return f"""
        SELECT s.*, {extra}
        FROM read_parquet({_quoted_path(source)}) AS s
        LEFT JOIN era5_{kind} AS e ON {condition}
    """


def align_city_era5(config_path: Path) -> dict[str, Any]:
    """Write independent ERA5-enriched SUHI panels and a coverage report."""
    paths, minimum_season, minimum_year, require_complete = _load_settings(config_path)
    output_dir = paths["output_dir"]
    sources = {kind: paths[f"{kind}_suhi_file"] for kind in KEYS}
    outputs = {
        kind: output_dir / f"suhi_{kind}_era5.parquet"
        for kind in KEYS
    }
    if set(outputs.values()) & set(sources.values()):
        raise ValueError("An output path would overwrite a SUHI input.")
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(":memory:")
    try:
        files = _load_era5(connection, paths["era5_raw_dir"])
        source_counts = {
            kind: _validate_suhi(connection, kind, path)
            for kind, path in sources.items()
        }
        _build_aggregates(connection, minimum_season, minimum_year)
        era5_summary = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT city_id),
                   MIN(printf('%04d-%02d', year, month)),
                   MAX(printf('%04d-%02d', year, month)),
                   COUNT(temperature_2m_c)
            FROM era5_monthly
            """
        ).fetchone()
        if era5_summary[0] == 0:
            raise ValueError("ERA5 CSV files contain no rows.")
        missing_months = connection.execute(
            f"""
            SELECT s.year, s.month, COUNT(*) AS missing_city_months
            FROM read_parquet({_quoted_path(sources['monthly'])}) s
            ANTI JOIN era5_monthly e
              ON s.ID_UC_G0 = e.city_id AND s.year = e.year AND s.month = e.month
            GROUP BY s.year, s.month ORDER BY s.year, s.month
            """
        ).fetchall()
        missing_city_months = sum(row[2] for row in missing_months)
        if require_complete and missing_city_months:
            missing_labels = [f"{year:04d}-{month:02d}" for year, month, _ in missing_months]
            missing_years = sorted({year for year, _, _ in missing_months})
            raise ValueError(
                f"ERA5 input lacks {missing_city_months:,} SUHI city-month records "
                f"across {len(missing_labels)} months in years {missing_years}; "
                "first missing months: "
                f"{', '.join(missing_labels[:12])}. Add the missing ERA5 CSVs "
                "before building complete outputs. Masked temperature values "
                "are allowed when the city-month record exists."
            )
        era5_unmatched = connection.execute(
            f"""
            SELECT COUNT(*) FROM era5_monthly e
            ANTI JOIN read_parquet({_quoted_path(sources['monthly'])}) s
              ON s.ID_UC_G0 = e.city_id AND s.year = e.year AND s.month = e.month
            """
        ).fetchone()[0]
        report: dict[str, Any] = {
            "era5_csv_files": [str(path) for path in files],
            "era5_source": {
                "rows": era5_summary[0],
                "cities": era5_summary[1],
                "first_month": era5_summary[2],
                "last_month": era5_summary[3],
                "valid_temperatures": era5_summary[4],
                "missing_temperatures": era5_summary[0] - era5_summary[4],
                "unmatched_city_months": era5_unmatched,
                "missing_suhi_city_month_records": missing_city_months,
                "months_with_missing_suhi_city_month_records": len(missing_months),
            },
            "aggregation": {
                "require_complete_city_months": require_complete,
                "weighting": "equal weight for each valid ERA5 calendar month",
                "season_year": "December belongs to the following DJF year",
                "minimum_valid_months_per_season": minimum_season,
                "minimum_valid_months_per_year": minimum_year,
                "modis_qc_applied_to_era5": False,
            },
            "outputs": {},
        }
        print(f"[ERA5] {report['era5_source']}", flush=True)
        print(f"[ERA5] Input files: {len(files)}", flush=True)
        with tempfile.TemporaryDirectory(prefix=".era5_alignment_", dir=output_dir) as temp:
            temp_dir = Path(temp)
            for kind, source in sources.items():
                minimum = minimum_season if kind == "seasonal" else minimum_year
                query = _aligned_query(kind, source, minimum)
                staged = temp_dir / outputs[kind].name
                print(f"[BUILD] {kind}: {source_counts[kind]:,} source rows", flush=True)
                connection.execute(
                    f"COPY ({query}) TO {_quoted_path(staged)} "
                    "(FORMAT PARQUET, COMPRESSION ZSTD)"
                )
                counts = connection.execute(
                    """
                    SELECT COUNT(*), COUNT(era5_temperature_2m_c)
                    FROM read_parquet(?)
                    """,
                    [str(staged)],
                ).fetchone()
                if counts[0] != source_counts[kind]:
                    raise ValueError(f"{kind} output row count differs from source.")
                columns = [
                    row[0] for row in connection.execute(
                        "DESCRIBE SELECT * FROM read_parquet(?)", [str(staged)]
                    ).fetchall()
                ]
                report["outputs"][kind] = {
                    "path": str(outputs[kind]),
                    "source": str(source),
                    "rows": counts[0],
                    "valid_era5_temperatures": counts[1],
                    "missing_era5_temperatures": counts[0] - counts[1],
                    "columns": columns,
                }
                print(f"[CHECK] {kind}: {counts[1]:,}/{counts[0]:,} ERA5 values", flush=True)
            staged_report = temp_dir / "era5_alignment_report.json"
            staged_report.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            for kind, output in outputs.items():
                (temp_dir / output.name).replace(output)
                print(f"[OUTPUT] {output}", flush=True)
            report_path = output_dir / "era5_alignment_report.json"
            staged_report.replace(report_path)
            print(f"[OUTPUT] {report_path}", flush=True)
        return report
    finally:
        connection.close()

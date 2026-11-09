"""Build city/season/slot OLS trends from seasonal SUHI data."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


SEASONS = ("DJF", "MAM", "JJA", "SON")
SLOTS = (
    "terra_day",
    "terra_night",
    "aqua_day",
    "aqua_night",
)


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
        raise ValueError("Config must be a JSON object.")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Config must contain a 'paths' object.")
    for key in ("seasonal_suhi_file", "output_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config 'paths.{key}' must be a non-empty path.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths[key] = path.resolve().as_posix()

    years = config.get("years")
    if not isinstance(years, dict):
        raise ValueError("Config must contain a 'years' object.")
    start = years.get("start")
    end = years.get("end")
    minimum = years.get("minimum_valid_years")
    for name, value in (("start", start), ("end", end)):
        if type(value) is not int or value < 1:
            raise ValueError(f"Config 'years.{name}' must be a positive integer.")
    if start > end:
        raise ValueError("Config 'years.start' must not exceed 'years.end'.")
    if type(minimum) is not int or not 2 <= minimum <= end - start + 1:
        raise ValueError(
            "Config 'years.minimum_valid_years' must be between 2 and the "
            "configured period length."
        )
    scale = config.get("beta_scale_years")
    if type(scale) is not int or scale < 1:
        raise ValueError("Config 'beta_scale_years' must be a positive integer.")
    config["_config_path"] = str(config_path)
    config["years"] = {
        "start": start,
        "end": end,
        "minimum_valid_years": minimum,
    }
    return config


def _quote_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _source_columns() -> set[str]:
    required = {
        "ID_UC_G0",
        "season_year",
        "season",
        "centroid_lat",
        "koppen_geiger_class",
    }
    for slot in SLOTS:
        required.update(
            {
                f"qc_pass_season_{slot}",
                f"urban_{slot}_mean",
                f"rural_{slot}_mean",
                f"suhi_{slot}_mean",
            }
        )
    return required


def _validate_source(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
) -> None:
    description = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [source.as_posix()],
    ).fetchall()
    available = {row[0] for row in description}
    missing = _source_columns() - available
    if missing:
        raise ValueError(
            f"Seasonal SUHI table is missing fields: {sorted(missing)}."
        )

    duplicates = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT ID_UC_G0, season_year, season
            FROM read_parquet(?)
            GROUP BY ID_UC_G0, season_year, season
            HAVING COUNT(*) > 1
        )
        """,
        [source.as_posix()],
    ).fetchone()[0]
    if duplicates:
        raise ValueError(
            "Seasonal SUHI table contains duplicate city/season_year/season keys."
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
        )
        """,
        [source.as_posix()],
    ).fetchone()[0]
    if inconsistent_metadata:
        raise ValueError(
            "Seasonal SUHI table has inconsistent latitude or climate "
            "metadata within a city."
        )

    invalid_qc_types = [
        row[0]
        for row in description
        if row[0] in {f"qc_pass_season_{slot}" for slot in SLOTS}
        and row[1] != "BOOLEAN"
    ]
    if invalid_qc_types:
        raise TypeError(
            f"Season QC columns must be BOOLEAN: {sorted(invalid_qc_types)}."
        )

    invalid_seasons = connection.execute(
        """
        SELECT DISTINCT season
        FROM read_parquet(?)
        WHERE season IS NULL
           OR season NOT IN ('DJF', 'MAM', 'JJA', 'SON')
        """,
        [source.as_posix()],
    ).fetchall()
    if invalid_seasons:
        raise ValueError(
            "Seasonal SUHI table contains unsupported seasons: "
            f"{sorted(str(row[0]) for row in invalid_seasons)}."
        )


def _trend_query(
    source: Path,
    start_year: int,
    end_year: int,
    minimum_valid_years: int,
    beta_scale_years: int,
) -> str:
    slot_rows = []
    for slot in SLOTS:
        slot_rows.append(
            "("
            f"{_quote_string(slot)}, "
            f"qc_pass_season_{slot}, "
            f"CAST(urban_{slot}_mean AS DOUBLE), "
            f"CAST(rural_{slot}_mean AS DOUBLE), "
            f"CAST(suhi_{slot}_mean AS DOUBLE)"
            ")"
        )
    slot_values = ",\n                ".join(slot_rows)
    source_path = _quote_string(source.as_posix())
    return f"""
        WITH seasonal AS (
            SELECT
                CAST(ID_UC_G0 AS BIGINT) AS ID_UC_G0,
                CAST(season_year AS INTEGER) AS season_year,
                CAST(season AS VARCHAR) AS season,
                CAST(centroid_lat AS DOUBLE) AS centroid_lat,
                CAST(koppen_geiger_class AS VARCHAR) AS climate,
                {", ".join(
                    f"CAST({column} AS DOUBLE) AS {column}"
                    for slot in SLOTS
                    for column in (
                        f"urban_{slot}_mean",
                        f"rural_{slot}_mean",
                        f"suhi_{slot}_mean",
                    )
                )},
                {", ".join(f"qc_pass_season_{slot}" for slot in SLOTS)}
            FROM read_parquet({source_path})
            WHERE season_year BETWEEN {start_year} AND {end_year}
        ),
        observations AS (
            SELECT
                seasonal.ID_UC_G0,
                seasonal.climate,
                seasonal.centroid_lat,
                CASE
                    WHEN seasonal.centroid_lat > 0 THEN 'north'
                    WHEN seasonal.centroid_lat < 0 THEN 'south'
                    WHEN seasonal.centroid_lat = 0 THEN 'equator'
                    ELSE NULL
                END AS hemisphere,
                seasonal.season,
                slot_values.slot,
                seasonal.season_year,
                slot_values.urban,
                slot_values.rural,
                slot_values.suhi
            FROM seasonal
            CROSS JOIN LATERAL (
                VALUES
                {slot_values}
            ) AS slot_values(slot, qc_pass, urban, rural, suhi)
            WHERE slot_values.qc_pass IS TRUE
              AND slot_values.urban IS NOT NULL
              AND slot_values.rural IS NOT NULL
              AND slot_values.suhi IS NOT NULL
              AND isfinite(slot_values.urban)
              AND isfinite(slot_values.rural)
              AND isfinite(slot_values.suhi)
        ),
        fitted AS (
            SELECT
                ID_UC_G0,
                climate,
                centroid_lat,
                hemisphere,
                season,
                slot,
                COUNT(*) AS n_valid_years,
                regr_slope(suhi, season_year) AS slope_suhi,
                regr_slope(urban, season_year) AS slope_urban,
                regr_slope(rural, season_year) AS slope_rural,
                regr_sxx(suhi, season_year) AS sxx_suhi,
                regr_sxx(urban, season_year) AS sxx_urban,
                regr_sxx(rural, season_year) AS sxx_rural,
                regr_sxy(suhi, season_year) AS sxy_suhi,
                regr_sxy(urban, season_year) AS sxy_urban,
                regr_sxy(rural, season_year) AS sxy_rural,
                regr_syy(suhi, season_year) AS syy_suhi,
                regr_syy(urban, season_year) AS syy_urban,
                regr_syy(rural, season_year) AS syy_rural
            FROM observations
            GROUP BY ID_UC_G0, climate, centroid_lat, hemisphere, season, slot
            HAVING COUNT(*) >= {minimum_valid_years}
        )
        SELECT
            ID_UC_G0,
            climate,
            centroid_lat,
            hemisphere,
            season,
            slot,
            slope_suhi * {beta_scale_years} AS beta_suhi,
            slope_urban * {beta_scale_years} AS beta_urban,
            slope_rural * {beta_scale_years} AS beta_rural,
            SQRT(
                GREATEST(
                    syy_suhi - sxy_suhi * sxy_suhi
                        / NULLIF(sxx_suhi, 0),
                    0
                ) / (n_valid_years - 2) / NULLIF(sxx_suhi, 0)
            ) * {beta_scale_years} AS se_suhi,
            SQRT(
                GREATEST(
                    syy_urban - sxy_urban * sxy_urban
                        / NULLIF(sxx_urban, 0),
                    0
                ) / (n_valid_years - 2) / NULLIF(sxx_urban, 0)
            ) * {beta_scale_years} AS se_urban,
            SQRT(
                GREATEST(
                    syy_rural - sxy_rural * sxy_rural
                        / NULLIF(sxx_rural, 0),
                    0
                ) / (n_valid_years - 2) / NULLIF(sxx_rural, 0)
            ) * {beta_scale_years} AS se_rural,
            n_valid_years
        FROM fitted
        ORDER BY ID_UC_G0, season, slot
    """


def build_city_season_slot_trends(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = Path(config["paths"]["seasonal_suhi_file"])
    if not source.is_file():
        raise FileNotFoundError(f"Seasonal SUHI table not found: {source}")

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
        _validate_source(connection, source)
        query = _trend_query(
            source,
            config["years"]["start"],
            config["years"]["end"],
            config["years"]["minimum_valid_years"],
            config["beta_scale_years"],
        )
        parquet_path = staging / "city_season_slot_trends.parquet"
        connection.execute(
            f"COPY ({query}) TO {_quote_string(parquet_path.as_posix())} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        row_count = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [parquet_path.as_posix()],
        ).fetchone()[0]
        if row_count == 0:
            raise ValueError(
                "No city/season/slot groups met the QC and valid-year "
                "requirements."
            )
        duplicate_groups = connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT ID_UC_G0, season, slot
                FROM read_parquet(?)
                GROUP BY ID_UC_G0, season, slot
                HAVING COUNT(*) > 1
            )
            """,
            [parquet_path.as_posix()],
        ).fetchone()[0]
        if duplicate_groups:
            raise ValueError(
                "Seasonal trend output has duplicate city/season/slot keys."
            )
        output_schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [parquet_path.as_posix()],
        ).fetchall()

        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source.name,
            "source_sha256": _sha256(source),
            "config_file": config_path.name,
            "config_sha256": _sha256(config_path),
            "years": config["years"],
            "period_column": "season_year",
            "season_definition": (
                "Fixed DJF/MAM/JJA/SON; December is assigned to the next "
                "season_year in the source seasonal table."
            ),
            "beta_scale_years": config["beta_scale_years"],
            "beta_units": (
                f"degrees_Celsius_per_{config['beta_scale_years']}_years"
            ),
            "standard_error_units": (
                f"degrees_Celsius_per_{config['beta_scale_years']}_years"
            ),
            "qc_policy": (
                "Use only observations whose qc_pass_season_<slot> flag is "
                "TRUE and whose urban, rural, and SUHI means are finite."
            ),
            "output_rows": row_count,
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
            f"[DONE] Wrote {row_count:,} city/season/slot trends to "
            f"{output_dir / parquet_path.name}",
            flush=True,
        )
        return output_dir / parquet_path.name
    finally:
        connection.close()
        if not published and staging.exists():
            shutil.rmtree(staging)

"""Compare paired city-level SUHI trends between selected calendar months."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


COMPARISONS = (
    {
        "climate_group": "D",
        "month_early": 2,
        "month_late": 7,
    },
    {
        "climate_group": "C",
        "month_early": 4,
        "month_late": 7,
    },
)
SLOTS = ("terra_day", "aqua_day")
REQUIRED_COLUMNS = {
    "ID_UC_G0",
    "climate",
    "month",
    "slot",
    "beta_suhi",
}


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
    for key in ("trend_file", "output_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config 'paths.{key}' must be a non-empty path.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths[key] = path.resolve().as_posix()
    config["_config_path"] = str(config_path)
    return config


def _validate_trends(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Trend table is missing fields: {sorted(missing)}.")
    if frame.empty:
        raise ValueError("Trend table contains no city/month/slot rows.")
    if frame["ID_UC_G0"].isna().any():
        raise ValueError("Trend table contains null city IDs.")
    if not frame["month"].between(1, 12).all():
        raise ValueError("Trend table contains invalid month values.")
    if frame.duplicated(["ID_UC_G0", "month", "slot"]).any():
        raise ValueError("Trend table has duplicate city/month/slot rows.")
    values = pd.to_numeric(frame["beta_suhi"], errors="raise")
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("Trend table contains non-finite SUHI trend slopes.")
    climate = frame["climate"].astype("string").str.strip().str.upper()
    frame = frame.copy()
    frame["climate_group"] = climate.str[0]
    return frame


def calculate_paired_month_deltas(
    trends: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return city-level paired deltas and their four requested summaries.

    Delta is defined as the later-month city slope minus the earlier-month
    city slope. The positive fraction uses every paired city as its denominator,
    including any exact zero deltas.
    """
    trends = _validate_trends(trends)
    paired_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    for comparison in COMPARISONS:
        climate_group = comparison["climate_group"]
        month_early = comparison["month_early"]
        month_late = comparison["month_late"]
        for slot in SLOTS:
            eligible = trends.loc[
                trends["climate_group"].eq(climate_group)
                & trends["slot"].eq(slot)
            ]
            early = eligible.loc[
                eligible["month"].eq(month_early),
                ["ID_UC_G0", "climate", "beta_suhi"],
            ].rename(
                columns={
                    "climate": "climate_early",
                    "beta_suhi": "beta_suhi_early",
                }
            )
            late = eligible.loc[
                eligible["month"].eq(month_late),
                ["ID_UC_G0", "climate", "beta_suhi"],
            ].rename(
                columns={
                    "climate": "climate_late",
                    "beta_suhi": "beta_suhi_late",
                }
            )
            paired = early.merge(
                late,
                on="ID_UC_G0",
                how="inner",
                validate="one_to_one",
            )
            climate_mismatch = (
                paired["climate_early"].fillna("").astype(str)
                != paired["climate_late"].fillna("").astype(str)
            )
            if climate_mismatch.any():
                examples = paired.loc[
                    climate_mismatch, "ID_UC_G0"
                ].head(10).tolist()
                raise ValueError(
                    "A city's Köppen climate changed between the compared "
                    f"months; city IDs include {examples}."
                )
            paired["climate_group"] = climate_group
            paired["month_early"] = month_early
            paired["month_late"] = month_late
            paired["slot"] = slot
            paired["delta_beta_suhi"] = (
                paired["beta_suhi_late"] - paired["beta_suhi_early"]
            )
            paired = paired.loc[
                :,
                [
                    "ID_UC_G0",
                    "climate_group",
                    "climate_early",
                    "month_early",
                    "month_late",
                    "slot",
                    "beta_suhi_early",
                    "beta_suhi_late",
                    "delta_beta_suhi",
                ],
            ]
            paired_records.extend(paired.to_dict("records"))

            deltas = paired["delta_beta_suhi"].to_numpy(dtype="float64")
            pair_count = int(deltas.size)
            positive_count = int(np.count_nonzero(deltas > 0))
            zero_count = int(np.count_nonzero(deltas == 0))
            negative_count = int(np.count_nonzero(deltas < 0))
            summary_records.append(
                {
                    "climate_group": climate_group,
                    "month_early": month_early,
                    "month_late": month_late,
                    "delta_definition": (
                        f"beta_suhi_{month_late:02d} - "
                        f"beta_suhi_{month_early:02d}"
                    ),
                    "slot": slot,
                    "n_paired_cities": pair_count,
                    "median_delta_beta_suhi": (
                        float(np.median(deltas)) if pair_count else np.nan
                    ),
                    "n_positive": positive_count,
                    "n_zero": zero_count,
                    "n_negative": negative_count,
                    "positive_fraction": (
                        positive_count / pair_count if pair_count else np.nan
                    ),
                }
            )

    paired_columns = [
        "ID_UC_G0",
        "climate_group",
        "climate_early",
        "month_early",
        "month_late",
        "slot",
        "beta_suhi_early",
        "beta_suhi_late",
        "delta_beta_suhi",
    ]
    summary_columns = [
        "climate_group",
        "month_early",
        "month_late",
        "delta_definition",
        "slot",
        "n_paired_cities",
        "median_delta_beta_suhi",
        "n_positive",
        "n_zero",
        "n_negative",
        "positive_fraction",
    ]
    paired_frame = pd.DataFrame.from_records(
        paired_records, columns=paired_columns
    ).sort_values(
        ["climate_group", "slot", "ID_UC_G0"],
        kind="stable",
        ignore_index=True,
    )
    summary_frame = pd.DataFrame.from_records(
        summary_records, columns=summary_columns
    )
    return paired_frame, summary_frame


def build_paired_month_city_trends(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = Path(config["paths"]["trend_file"])
    if not source.is_file():
        raise FileNotFoundError(f"City-month trend table not found: {source}")
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
    try:
        connection = duckdb.connect(":memory:")
        try:
            trends = connection.execute(
                "SELECT * FROM read_parquet(?)",
                [source.as_posix()],
            ).df()
        finally:
            connection.close()

        paired, summary = calculate_paired_month_deltas(trends)
        paired_path = staging / "paired_month_delta_city_level.csv"
        summary_path = staging / "paired_month_delta_summary.csv"
        paired.to_csv(paired_path, index=False, float_format="%.10g")
        summary.to_csv(summary_path, index=False, float_format="%.10g")
        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trend_file": source.name,
            "trend_sha256": _sha256(source),
            "config_file": config_path.name,
            "config_sha256": _sha256(config_path),
            "delta_definition": "later month beta_suhi minus earlier month beta_suhi",
            "positive_fraction_denominator": (
                "All cities with valid trend rows for both compared months "
                "and the same sensor slot, including zero deltas."
            ),
            "comparisons": list(COMPARISONS),
            "slots": list(SLOTS),
            "paired_city_rows": len(paired),
            "summary_rows": len(summary),
            "outputs": [paired_path.name, summary_path.name],
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
            "[DONE] Paired-month summaries published at "
            f"{output_dir / summary_path.name}",
            flush=True,
        )
        return output_dir
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

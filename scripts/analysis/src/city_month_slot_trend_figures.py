"""Summarize and plot climate-specific city/month/slot trend distributions."""
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
SLOTS = (
    ("terra_day", "Terra Day", "#D55E00", "o"),
    ("terra_night", "Terra Night", "#E69F00", "s"),
    ("aqua_day", "Aqua Day", "#0072B2", "o"),
    ("aqua_night", "Aqua Night", "#56B4E9", "s"),
)
METRICS = (
    ("beta_urban", "Urban (βU)", "#D55E00"),
    ("beta_rural", "Rural (βR)", "#0072B2"),
    ("beta_suhi", "SUHI (βSUHI)", "#009E73"),
)
REQUIRED_COLUMNS = {
    "ID_UC_G0",
    "climate",
    "centroid_lat",
    "hemisphere",
    "month",
    "slot",
    "beta_suhi",
    "beta_urban",
    "beta_rural",
    "se_suhi",
    "se_urban",
    "se_rural",
    "n_valid_years",
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

    climate_groups = config.get("climate_groups")
    if (
        not isinstance(climate_groups, list)
        or not climate_groups
        or any(
            not isinstance(group, str)
            or len(group) != 1
            or group.upper() not in {"A", "B", "C", "D", "E"}
            for group in climate_groups
        )
        or len({group.upper() for group in climate_groups}) != len(climate_groups)
    ):
        raise ValueError(
            "Config 'climate_groups' must be unique Köppen major groups "
            "from A, B, C, D, and E."
        )
    config["climate_groups"] = [group.upper() for group in climate_groups]

    hemisphere = config.get("hemisphere", "all")
    if not isinstance(hemisphere, str) or hemisphere.strip().lower() not in {
        "all",
        "north",
        "south",
    }:
        raise ValueError(
            "Config 'hemisphere' must be 'all', 'north', or 'south'."
        )
    config["hemisphere"] = hemisphere.strip().lower()

    dpi = config.get("dpi")
    if type(dpi) is not int or dpi < 72 or dpi > 1200:
        raise ValueError("Config 'dpi' must be an integer from 72 to 1200.")
    tolerance = config.get("decomposition_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError(
            "Config 'decomposition_tolerance' must be a finite non-negative number."
        )
    config["_config_path"] = str(config_path)
    return config


def _validate_trends(frame: pd.DataFrame, config: dict[str, Any]) -> float:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Trend table is missing fields: {sorted(missing)}.")
    if frame.empty:
        raise ValueError("Trend table contains no city/month/slot rows.")
    if frame.duplicated(["ID_UC_G0", "month", "slot"]).any():
        raise ValueError("Trend table has duplicate city/month/slot keys.")
    if frame["ID_UC_G0"].isna().any():
        raise ValueError("Trend table has null city IDs.")
    if not frame["month"].between(1, 12).all():
        raise ValueError("Trend table contains month values outside 1..12.")
    valid_slots = {slot for slot, _, _, _ in SLOTS}
    if not set(frame["slot"].dropna()).issubset(valid_slots):
        raise ValueError("Trend table contains unsupported slots.")
    if frame["centroid_lat"].isna().any():
        raise ValueError("Trend table contains rows without centroid latitude.")

    metric_columns = ["beta_urban", "beta_rural", "beta_suhi"]
    values = frame[metric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("Trend coefficients must all be finite.")
    errors = (
        values["beta_suhi"]
        - (values["beta_urban"] - values["beta_rural"])
    ).abs()
    max_error = float(errors.max())
    tolerance = float(config["decomposition_tolerance"])
    if max_error > tolerance:
        raise ValueError(
            "City-level trend decomposition failed: "
            f"max |beta_suhi - (beta_urban - beta_rural)|={max_error:.12g}, "
            f"tolerance={tolerance:.12g}."
        )
    frame.loc[:, metric_columns] = values
    climate_codes = frame["climate"].astype("string").str.strip().str.upper()
    missing_climate = climate_codes.isna() | climate_codes.isin(
        ["", "NODATA", "NO DATA", "NONE", "NULL"]
    )
    climate_groups = climate_codes.str[0].mask(missing_climate)
    invalid_climate = climate_groups.notna() & ~climate_groups.isin(
        ["A", "B", "C", "D", "E"]
    )
    if invalid_climate.any():
        examples = sorted(climate_codes.loc[invalid_climate].unique().tolist())
        raise ValueError(
            f"Trend table contains invalid Köppen climate labels: {examples[:10]}."
        )
    frame["climate_group"] = climate_groups
    return max_error


def _filter_trends_by_hemisphere(
    frame: pd.DataFrame,
    hemisphere: str,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    source_rows = len(frame)
    source_cities = int(frame["ID_UC_G0"].nunique())
    if hemisphere == "all":
        return frame, {
            "selection": "all",
            "source_rows": source_rows,
            "source_cities": source_cities,
            "selected_rows": source_rows,
            "selected_cities": source_cities,
            "excluded_rows": 0,
            "excluded_cities": 0,
            "excluded_north_rows": 0,
            "excluded_south_rows": 0,
            "excluded_equator_rows": 0,
            "excluded_missing_hemisphere_rows": 0,
        }

    labels = frame["hemisphere"].astype("string").str.strip().str.lower()
    invalid = labels.notna() & ~labels.isin({"north", "south", "equator"})
    if invalid.any():
        examples = sorted(labels.loc[invalid].unique().tolist())
        raise ValueError(
            f"Trend table contains invalid hemisphere labels: {examples[:10]}."
        )

    selected_mask = labels.eq(hemisphere).fillna(False)
    selected = frame.loc[selected_mask].copy()
    if selected.empty:
        raise ValueError(
            f"No city/month/slot trends matched hemisphere '{hemisphere}'."
        )
    excluded = frame.loc[~selected_mask]
    return selected, {
        "selection": hemisphere,
        "source_rows": source_rows,
        "source_cities": source_cities,
        "selected_rows": len(selected),
        "selected_cities": int(selected["ID_UC_G0"].nunique()),
        "excluded_rows": len(excluded),
        "excluded_cities": int(excluded["ID_UC_G0"].nunique()),
        "excluded_north_rows": int(labels.eq("north").sum())
        if hemisphere == "south"
        else 0,
        "excluded_south_rows": int(labels.eq("south").sum())
        if hemisphere == "north"
        else 0,
        "excluded_equator_rows": int(labels.eq("equator").sum()),
        "excluded_missing_hemisphere_rows": int(labels.isna().sum()),
    }


def _climate_exclusion_counts(
    frame: pd.DataFrame,
    climate_groups: list[str],
) -> dict[str, int]:
    missing_climate = frame["climate_group"].isna()
    outside_groups = (
        frame["climate_group"].notna()
        & ~frame["climate_group"].isin(climate_groups)
    )
    return {
        "unclassified_rows": int(missing_climate.sum()),
        "unclassified_cities": int(
            frame.loc[missing_climate, "ID_UC_G0"].nunique()
        ),
        "outside_configured_groups_rows": int(outside_groups.sum()),
        "outside_configured_groups_cities": int(
            frame.loc[outside_groups, "ID_UC_G0"].nunique()
        ),
    }


def summarize_trends(
    frame: pd.DataFrame,
    climate_groups: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if "climate_group" not in frame.columns:
        frame = frame.copy()
        frame["climate_group"] = (
            frame["climate"].astype("string").str[0].str.upper()
        )
    selected = frame.loc[frame["climate_group"].isin(climate_groups)]
    for climate in climate_groups:
        for slot, _, _, _ in SLOTS:
            for month in range(1, 13):
                group = selected.loc[
                    selected["climate_group"].eq(climate)
                    & selected["slot"].eq(slot)
                    & selected["month"].eq(month)
                ]
                record: dict[str, Any] = {
                    "climate_group": climate,
                    "month": month,
                    "slot": slot,
                    "n_cities": int(group["ID_UC_G0"].nunique()),
                }
                for metric, _, _ in METRICS:
                    values = group[metric].dropna()
                    record[f"median_{metric}"] = (
                        float(values.median()) if not values.empty else np.nan
                    )
                    record[f"q25_{metric}"] = (
                        float(values.quantile(0.25))
                        if not values.empty else np.nan
                    )
                    record[f"q75_{metric}"] = (
                        float(values.quantile(0.75))
                        if not values.empty else np.nan
                    )
                records.append(record)
    return pd.DataFrame.from_records(records)


def _plot_figure_one(
    summary: pd.DataFrame,
    climate_groups: list[str],
    hemisphere: str,
):
    figure, axes = plt.subplots(
        len(climate_groups),
        len(SLOTS),
        figsize=(19, max(3.2 * len(climate_groups), 8)),
        sharex=True,
        squeeze=False,
    )
    for row_index, climate in enumerate(climate_groups):
        for column_index, (slot, slot_label, color, marker) in enumerate(SLOTS):
            axis = axes[row_index, column_index]
            data = summary.loc[
                summary["climate_group"].eq(climate)
                & summary["slot"].eq(slot)
            ].sort_values("month")
            months = data["month"].to_numpy(dtype=int)
            median = data["median_beta_suhi"].to_numpy(dtype=float)
            q25 = data["q25_beta_suhi"].to_numpy(dtype=float)
            q75 = data["q75_beta_suhi"].to_numpy(dtype=float)
            counts = data["n_cities"].to_numpy(dtype=int)
            valid = np.isfinite(median)
            if valid.any():
                axis.errorbar(
                    months[valid],
                    median[valid],
                    yerr=np.vstack(
                        (
                            median[valid] - q25[valid],
                            q75[valid] - median[valid],
                        )
                    ),
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=1.2,
                    capsize=2,
                )
            axis.axhline(0, color="0.35", linewidth=0.7, alpha=0.7)
            axis.set_xlim(0.5, 12.5)
            axis.set_xticks(range(1, 13), MONTH_LABELS, rotation=45, ha="right")
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(f"{climate} climate — {slot_label}")
            if column_index == 0:
                axis.set_ylabel("βSUHI (°C/decade)")
            if row_index == len(climate_groups) - 1:
                axis.set_xlabel("Calendar month")
            for month, count in zip(months, counts):
                axis.text(
                    month,
                    0.015,
                    f"n={count}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="0.3",
                )
    figure.suptitle(
        "Figure 1. Climate-specific monthly SUHI trends\n"
        f"Hemisphere: {hemisphere}; median city slope with interquartile "
        "range; n = eligible cities",
        y=0.995,
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def _plot_figure_two(
    summary: pd.DataFrame,
    climate_groups: list[str],
    hemisphere: str,
):
    figure, axes = plt.subplots(
        len(climate_groups),
        len(SLOTS),
        figsize=(19, max(3.2 * len(climate_groups), 8)),
        sharex=True,
        squeeze=False,
    )
    offsets = (-0.22, 0.0, 0.22)
    for row_index, climate in enumerate(climate_groups):
        for column_index, (slot, slot_label, _, _) in enumerate(SLOTS):
            axis = axes[row_index, column_index]
            data = summary.loc[
                summary["climate_group"].eq(climate)
                & summary["slot"].eq(slot)
            ].sort_values("month")
            for offset, (metric, label, color) in zip(offsets, METRICS):
                median = data[f"median_{metric}"].to_numpy(dtype=float)
                q25 = data[f"q25_{metric}"].to_numpy(dtype=float)
                q75 = data[f"q75_{metric}"].to_numpy(dtype=float)
                months = data["month"].to_numpy(dtype=float)
                valid = np.isfinite(median)
                if valid.any():
                    axis.errorbar(
                        months[valid] + offset,
                        median[valid],
                        yerr=np.vstack(
                            (
                                median[valid] - q25[valid],
                                q75[valid] - median[valid],
                            )
                        ),
                        color=color,
                        marker="o",
                        markersize=3.5,
                        linewidth=1.0,
                        capsize=2,
                        label=label,
                    )
            axis.axhline(0, color="0.35", linewidth=0.7, alpha=0.7)
            axis.set_xlim(0.5, 12.5)
            axis.set_xticks(range(1, 13), MONTH_LABELS, rotation=45, ha="right")
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(f"{climate} climate — {slot_label}")
            if column_index == 0:
                axis.set_ylabel("Slope (°C/decade)")
            if row_index == len(climate_groups) - 1:
                axis.set_xlabel("Calendar month")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        "Figure 2. Urban–Rural trend decomposition\n"
        "City-level βSUHI = βU − βR is checked before summarization; "
        f"climate medians are shown independently; hemisphere: {hemisphere}",
        y=0.995,
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def build_city_month_slot_trend_figures(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = Path(config["paths"]["trend_file"])
    if not source.is_file():
        raise FileNotFoundError(f"City-month trend table not found: {source}")
    output_dir = Path(config["paths"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(
            f"Figure output directory already exists; refusing to overwrite: "
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
        frame = connection.execute(
            "SELECT * FROM read_parquet(?)",
            [source.as_posix()],
        ).df()
        frame, hemisphere_filter = _filter_trends_by_hemisphere(
            frame, config["hemisphere"]
        )
        print(
            f"[INFO] Hemisphere filter={hemisphere_filter['selection']}; "
            f"cities={hemisphere_filter['selected_cities']:,}/"
            f"{hemisphere_filter['source_cities']:,}; "
            f"rows={hemisphere_filter['selected_rows']:,}/"
            f"{hemisphere_filter['source_rows']:,}; "
            f"excluded rows={hemisphere_filter['excluded_rows']:,}.",
            flush=True,
        )
        max_decomposition_error = _validate_trends(frame, config)
        exclusions = _climate_exclusion_counts(
            frame, config["climate_groups"]
        )
        print(
            "[INFO] Climate grouping exclusions: "
            f"unclassified={exclusions['unclassified_rows']:,} rows/"
            f"{exclusions['unclassified_cities']:,} cities; "
            "outside configured groups="
            f"{exclusions['outside_configured_groups_rows']:,} rows/"
            f"{exclusions['outside_configured_groups_cities']:,} cities.",
            flush=True,
        )
        summary = summarize_trends(frame, config["climate_groups"])
        summary.to_csv(
            staging / "climate_month_slot_trend_summary.csv",
            index=False,
            float_format="%.8g",
        )

        figure_one = _plot_figure_one(
            summary, config["climate_groups"], config["hemisphere"]
        )
        figure_one_path = staging / "figure_1_climate_monthly_suhi_trends.png"
        figure_one.savefig(figure_one_path, dpi=config["dpi"], bbox_inches="tight")
        plt.close(figure_one)

        figure_two = _plot_figure_two(
            summary, config["climate_groups"], config["hemisphere"]
        )
        figure_two_path = staging / "figure_2_urban_rural_trend_decomposition.png"
        figure_two.savefig(figure_two_path, dpi=config["dpi"], bbox_inches="tight")
        plt.close(figure_two)

        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trend_file": source.name,
            "trend_sha256": _sha256(source),
            "config_file": config_path.name,
            "config_sha256": _sha256(config_path),
            "climate_grouping": "first character of Köppen-Geiger climate class",
            "climate_groups_included": config["climate_groups"],
            "hemisphere_filter": hemisphere_filter,
            "climate_grouping_exclusions": exclusions,
            "city_level_identity": "beta_suhi = beta_urban - beta_rural",
            "max_absolute_city_level_decomposition_error": (
                max_decomposition_error
            ),
            "decomposition_tolerance": config["decomposition_tolerance"],
            "summary_note": (
                "Urban, rural, and SUHI climate-group medians and IQRs are "
                "computed independently across cities and are not additive."
            ),
            "outputs": [
                "figure_1_climate_monthly_suhi_trends.png",
                "figure_2_urban_rural_trend_decomposition.png",
                "climate_month_slot_trend_summary.csv",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(
                f"Figure output directory appeared during build: {output_dir}"
            )
        staging.rename(output_dir)
        published = True
        print(f"[DONE] Figures and summary published at {output_dir}", flush=True)
        return output_dir
    finally:
        connection.close()
        if not published and staging.exists():
            import shutil

            shutil.rmtree(staging)

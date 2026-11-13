"""Plot seasonal gamma responses for stable, development, and overall zones."""
from __future__ import annotations

import json
import shutil
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
from matplotlib.lines import Line2D

from src.city_season_slot_trend_figures import SLOTS, _filter_trends_by_hemisphere
from src.city_season_subclimate_trend_figures import _select_subclimates
from src.city_seasonal_gamma_subclimate_figures import (
    _load_config,
    _publish_staged_outputs,
    _sha256,
)


URBAN_ZONES = ("stable", "development", "overall")
GAMMA_METRICS = (
    ("gamma_urban", "Urban (γU)", "#4C78A8"),
    ("gamma_rural", "Rural (γR)", "#F58518"),
    ("gamma_suhi", "SUHI (γSUHI)", "#54A24B"),
)
REQUIRED_COLUMNS = {
    "ID_UC_G0",
    "climate",
    "hemisphere",
    "urban_zone",
    "slot",
    "summer_season",
    "winter_season",
    "gamma_urban",
    "gamma_rural",
    "gamma_suhi",
}


def _validate_zone_gammas(
    frame: pd.DataFrame,
    tolerance: float,
) -> float:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Seasonal zone gamma table is missing fields: {sorted(missing)}."
        )
    if frame.empty:
        raise ValueError("Seasonal zone gamma table contains no rows.")
    if frame.duplicated(["ID_UC_G0", "urban_zone", "slot"]).any():
        raise ValueError(
            "Seasonal zone gamma table has duplicate city/zone/slot keys."
        )
    if frame["ID_UC_G0"].isna().any():
        raise ValueError("Seasonal zone gamma table contains null city IDs.")
    if set(frame["urban_zone"].dropna().unique()) != set(URBAN_ZONES):
        raise ValueError(
            "Seasonal zone gamma table must contain stable, development, "
            "and overall zones."
        )
    if frame["urban_zone"].isna().any():
        raise ValueError("Seasonal zone gamma table contains null urban zones.")
    if set(frame["hemisphere"].dropna().unique()) - {"north", "south"}:
        raise ValueError(
            "Seasonal zone gamma table must contain only north/south rows."
        )
    if frame["hemisphere"].isna().any():
        raise ValueError("Seasonal zone gamma table contains null hemispheres.")
    expected_seasons = frame["hemisphere"].map(
        {
            "north": ("JJA", "DJF"),
            "south": ("DJF", "JJA"),
        }
    )
    actual_seasons = list(zip(frame["summer_season"], frame["winter_season"]))
    if any(expected != actual for expected, actual in zip(expected_seasons, actual_seasons)):
        raise ValueError(
            "Seasonal zone gamma rows do not follow the hemisphere-specific "
            "summer/winter convention."
        )
    if set(frame["slot"].dropna().unique()) - {slot[0] for slot in SLOTS}:
        raise ValueError("Seasonal zone gamma table contains unsupported slots.")
    if frame["slot"].isna().any():
        raise ValueError("Seasonal zone gamma table contains null slots.")
    for metric, _, _ in GAMMA_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(
                f"Seasonal zone gamma table has non-finite {metric} values."
            )
        frame[metric] = values
    errors = frame["gamma_suhi"] - (
        frame["gamma_urban"] - frame["gamma_rural"]
    )
    max_error = float(errors.abs().max())
    if max_error > tolerance:
        raise ValueError(
            "Seasonal zone gamma decomposition failed: "
            f"max |gamma_suhi - (gamma_urban - gamma_rural)|="
            f"{max_error:.12g}, tolerance={tolerance:.12g}."
        )
    return max_error


def summarize_zone_gammas(
    frame: pd.DataFrame,
    subclasses: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for climate in subclasses:
        for urban_zone in URBAN_ZONES:
            for slot, _, _, _ in SLOTS:
                group = frame.loc[
                    frame["climate"].eq(climate)
                    & frame["urban_zone"].eq(urban_zone)
                    & frame["slot"].eq(slot)
                ]
                record: dict[str, Any] = {
                    "climate": climate,
                    "urban_zone": urban_zone,
                    "slot": slot,
                    "n_cities": int(group["ID_UC_G0"].nunique()),
                }
                for metric, _, _ in GAMMA_METRICS:
                    values = group[metric].dropna()
                    record[f"median_{metric}"] = (
                        float(values.median()) if not values.empty else np.nan
                    )
                    record[f"q25_{metric}"] = (
                        float(values.quantile(0.25))
                        if not values.empty
                        else np.nan
                    )
                    record[f"q75_{metric}"] = (
                        float(values.quantile(0.75))
                        if not values.empty
                        else np.nan
                    )
                records.append(record)
    return pd.DataFrame.from_records(records)


def _plot_suhi_summary(
    summary: pd.DataFrame,
    subclasses: list[str],
    major_climate_group: str,
    hemisphere: str,
):
    figure, axes = plt.subplots(
        len(subclasses),
        len(SLOTS),
        figsize=(19, max(3.0 * len(subclasses), 7)),
        squeeze=False,
        sharey=True,
    )
    zone_colors = {
        "stable": "#4C78A8",
        "development": "#F58518",
        "overall": "#54A24B",
    }
    positions = np.arange(len(URBAN_ZONES), dtype=float)
    for row_index, climate in enumerate(subclasses):
        for column_index, (slot, slot_label, _, _) in enumerate(SLOTS):
            axis = axes[row_index, column_index]
            for position, urban_zone in zip(positions, URBAN_ZONES):
                row = summary.loc[
                    summary["climate"].eq(climate)
                    & summary["urban_zone"].eq(urban_zone)
                    & summary["slot"].eq(slot)
                ].iloc[0]
                median = row["median_gamma_suhi"]
                if pd.notna(median):
                    axis.errorbar(
                        [position],
                        [median],
                        yerr=[
                            [median - row["q25_gamma_suhi"]],
                            [row["q75_gamma_suhi"] - median],
                        ],
                        color=zone_colors[urban_zone],
                        marker="o",
                        markersize=5,
                        linewidth=1.0,
                        capsize=3,
                    )
            axis.axhline(0, color="0.35", linewidth=0.7, alpha=0.7)
            axis.set_xticks(positions, ["Stable", "Development", "Overall"])
            axis.tick_params(axis="x", labelrotation=25)
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(f"{climate} — {slot_label}")
            if column_index == 0:
                axis.set_ylabel("γSUHI (°C/decade)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=zone_colors[zone],
            label=zone.title(),
        )
        for zone in URBAN_ZONES
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        "Summer-minus-winter SUHI trend by urban pixel zone\n"
        f"Major climate {major_climate_group}; hemisphere: {hemisphere}; "
        "city median and IQR",
        y=0.995,
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _plot_zone_synchrony(
    frame: pd.DataFrame,
    subclasses: list[str],
    major_climate_group: str,
    hemisphere: str,
):
    category_colors = dict(
        zip(subclasses, plt.get_cmap("tab20")(np.linspace(0, 1, len(subclasses))))
    )
    max_abs_value = max(
        float(frame["gamma_rural"].abs().max()),
        float(frame["gamma_urban"].abs().max()),
    )
    limit = max(max_abs_value * 1.05, 0.1)
    figure, axes = plt.subplots(
        len(URBAN_ZONES),
        len(SLOTS),
        figsize=(19, max(4.0 * len(URBAN_ZONES), 10)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row_index, urban_zone in enumerate(URBAN_ZONES):
        zone_frame = frame.loc[frame["urban_zone"].eq(urban_zone)]
        for column_index, (slot, slot_label, _, _) in enumerate(SLOTS):
            axis = axes[row_index, column_index]
            slot_frame = zone_frame.loc[zone_frame["slot"].eq(slot)]
            for climate in subclasses:
                category_rows = slot_frame.loc[slot_frame["climate"].eq(climate)]
                if category_rows.empty:
                    continue
                axis.scatter(
                    category_rows["gamma_rural"],
                    category_rows["gamma_urban"],
                    color=category_colors[climate],
                    s=14,
                    alpha=0.65,
                    linewidths=0,
                    label="_nolegend_",
                )
            axis.plot(
                [-limit, limit],
                [-limit, limit],
                color="0.2",
                linestyle="--",
                linewidth=1.0,
            )
            axis.set_xlim(-limit, limit)
            axis.set_ylim(-limit, limit)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.set_title(f"{urban_zone.title()} — {slot_label} (n={len(slot_frame):,})")
            if row_index == len(URBAN_ZONES) - 1:
                axis.set_xlabel("γR (°C/decade)")
            if column_index == 0:
                axis.set_ylabel("γU (°C/decade)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=category_colors[climate],
            markeredgecolor="none",
            label=climate,
        )
        for climate in subclasses
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle="--",
            linewidth=1.0,
            label="y = x (γSUHI = 0)",
        )
    )
    figure.suptitle(
        "City-level seasonal trend synchrony by urban pixel zone\n"
        f"Major climate {major_climate_group}; hemisphere: {hemisphere}; "
        "colored by Köppen subclass",
        y=0.995,
        fontsize=14,
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    return figure


def build_city_season_zone_gamma_figures(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = Path(config["paths"]["gamma_file"])
    if not source.is_file():
        raise FileNotFoundError(f"City seasonal zone gamma table not found: {source}")
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}_staging_",
            dir=output_dir.parent,
        )
    )
    connection = duckdb.connect(":memory:")
    try:
        frame = connection.execute(
            "SELECT * FROM read_parquet(?)", [source.as_posix()]
        ).df()
        max_decomposition_error = _validate_zone_gammas(
            frame, config["decomposition_tolerance"]
        )
        frame, hemisphere_filter = _filter_trends_by_hemisphere(
            frame, config["hemisphere"]
        )
        selected, subclasses, climate_counts = _select_subclimates(
            frame,
            config["major_climate_group"],
            config["subclimate_classes"],
        )
        if selected.empty:
            raise ValueError("No city/zone gamma rows remain for plotting.")
        if set(selected["urban_zone"].unique()) != set(URBAN_ZONES):
            raise ValueError(
                "The climate/hemisphere selection must contain all three "
                "urban zones."
            )

        summary = summarize_zone_gammas(selected, subclasses)
        summary_path = staging / "city_season_zone_gamma_summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.8g")

        suhi_figure = _plot_suhi_summary(
            summary,
            subclasses,
            config["major_climate_group"],
            config["hemisphere"],
        )
        suhi_path = staging / "figure_1_zone_gamma_suhi_summary.png"
        suhi_figure.savefig(suhi_path, dpi=config["dpi"], bbox_inches="tight")
        plt.close(suhi_figure)

        synchrony_figure = _plot_zone_synchrony(
            selected,
            subclasses,
            config["major_climate_group"],
            config["hemisphere"],
        )
        synchrony_path = staging / "figure_2_zone_city_synchrony.png"
        synchrony_figure.savefig(
            synchrony_path, dpi=config["dpi"], bbox_inches="tight"
        )
        plt.close(synchrony_figure)

        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gamma_file": source.name,
            "gamma_sha256": _sha256(source),
            "config_file": config_path.name,
            "config_sha256": _sha256(config_path),
            "major_climate_group": config["major_climate_group"],
            "subclimate_classes": subclasses,
            "hemisphere_filter": hemisphere_filter,
            "climate_group_counts": climate_counts,
            "urban_zones": list(URBAN_ZONES),
            "decomposition_tolerance": config["decomposition_tolerance"],
            "summer_minus_winter_direction": {
                "north": "JJA - DJF",
                "south": "DJF - JJA",
            },
            "city_level_identity": "gamma_suhi = gamma_urban - gamma_rural",
            "max_absolute_city_level_decomposition_error": (
                max_decomposition_error
            ),
            "outputs": [
                suhi_path.name,
                synchrony_path.name,
                summary_path.name,
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish_staged_outputs(staging, output_dir)
        print(
            f"[DONE] Seasonal urban-zone gamma figures published at {output_dir}",
            flush=True,
        )
        return output_dir
    finally:
        connection.close()
        if staging.exists():
            shutil.rmtree(staging)

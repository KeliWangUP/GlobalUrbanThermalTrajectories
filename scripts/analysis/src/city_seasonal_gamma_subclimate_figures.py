"""Plot summer-minus-winter trend contrasts by Köppen subclass."""
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.city_season_slot_trend_figures import (
    METRICS as BETA_METRICS,
    SLOTS,
    _filter_trends_by_hemisphere,
)
from src.city_season_subclimate_trend_figures import _select_subclimates


GAMMA_METRICS = (
    ("gamma_urban", "Urban (γU)", BETA_METRICS[0][2]),
    ("gamma_rural", "Rural (γR)", BETA_METRICS[1][2]),
    ("gamma_suhi", "SUHI (γSUHI)", BETA_METRICS[2][2]),
)
REQUIRED_COLUMNS = {
    "ID_UC_G0",
    "climate",
    "hemisphere",
    "slot",
    "summer_season",
    "winter_season",
    "gamma_suhi",
    "gamma_urban",
    "gamma_rural",
    "GC_DEV_USR",
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
    for key in ("gamma_file", "output_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config 'paths.{key}' must be a non-empty path.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths[key] = path.resolve().as_posix()

    major = config.get("major_climate_group")
    if not isinstance(major, str) or major.strip().upper() not in {
        "A",
        "B",
        "C",
        "D",
        "E",
    }:
        raise ValueError(
            "Config 'major_climate_group' must be one of A, B, C, D, or E."
        )
    config["major_climate_group"] = major.strip().upper()

    subclasses = config.get("subclimate_classes", [])
    if (
        not isinstance(subclasses, list)
        or any(
            not isinstance(subclass, str)
            or len(subclass.strip()) < 2
            or subclass.strip()[0].upper() != config["major_climate_group"]
            for subclass in subclasses
        )
    ):
        raise ValueError(
            "Config 'subclimate_classes' must be a list of Köppen subclasses "
            "belonging to the selected major group."
        )
    normalized_subclasses = [item.strip().upper() for item in subclasses]
    if len(set(normalized_subclasses)) != len(normalized_subclasses):
        raise ValueError("Config 'subclimate_classes' must not contain duplicates.")
    config["subclimate_classes"] = normalized_subclasses

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
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError(
            "Config 'decomposition_tolerance' must be a finite non-negative number."
        )
    config["decomposition_tolerance"] = float(tolerance)
    return config


def _validate_gamma_trends(
    frame: pd.DataFrame,
    tolerance: float,
) -> float:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Seasonal gamma table is missing fields: {sorted(missing)}."
        )
    if frame.empty:
        raise ValueError("Seasonal gamma table contains no rows.")
    if frame.duplicated(["ID_UC_G0", "slot"]).any():
        raise ValueError("Seasonal gamma table has duplicate city/slot keys.")
    if frame["ID_UC_G0"].isna().any():
        raise ValueError("Seasonal gamma table contains null city IDs.")
    if frame["GC_DEV_USR"].isna().any() or (
        frame["GC_DEV_USR"].astype("string").str.strip().eq("").any()
    ):
        raise ValueError(
            "Seasonal gamma table contains missing GC_DEV_USR regions."
        )
    if frame["slot"].isna().any():
        raise ValueError("Seasonal gamma table contains null observation slots.")
    valid_slots = {slot for slot, _, _, _ in SLOTS}
    if not set(frame["slot"].dropna()).issubset(valid_slots):
        raise ValueError("Seasonal gamma table contains unsupported slots.")
    valid_hemispheres = {"north", "south"}
    hemisphere = frame["hemisphere"].astype("string").str.strip().str.lower()
    if hemisphere.isna().any() or not set(hemisphere).issubset(
        valid_hemispheres
    ):
        raise ValueError(
            "Seasonal gamma table must contain only north/south hemisphere rows."
        )
    hemisphere_labels = (
        frame["hemisphere"].astype("string").str.strip().str.lower()
    )
    direction_is_valid = (
        (
            hemisphere_labels.eq("north")
            & frame["summer_season"].eq("JJA")
            & frame["winter_season"].eq("DJF")
        )
        | (
            hemisphere_labels.eq("south")
            & frame["summer_season"].eq("DJF")
            & frame["winter_season"].eq("JJA")
        )
    )
    if not direction_is_valid.all():
        raise ValueError(
            "Seasonal gamma rows do not follow the hemisphere-specific "
            "summer/winter season definitions."
        )

    metric_columns = [metric for metric, _, _ in GAMMA_METRICS]
    values = frame[metric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("Seasonal gamma coefficients must all be finite.")
    errors = (
        values["gamma_suhi"]
        - (values["gamma_urban"] - values["gamma_rural"])
    ).abs()
    max_error = float(errors.max())
    if max_error > tolerance:
        raise ValueError(
            "City-level seasonal gamma decomposition failed: "
            f"max |gamma_suhi - (gamma_urban - gamma_rural)|="
            f"{max_error:.12g}, tolerance={tolerance:.12g}."
        )
    frame.loc[:, metric_columns] = values
    return max_error


def summarize_subclimate_gammas(
    frame: pd.DataFrame,
    subclasses: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for climate in subclasses:
        for slot, _, _, _ in SLOTS:
            group = frame.loc[
                frame["climate"].eq(climate) & frame["slot"].eq(slot)
            ]
            record: dict[str, Any] = {
                "climate": climate,
                "slot": slot,
                "n_cities": int(group["ID_UC_G0"].nunique()),
            }
            for metric, _, _ in GAMMA_METRICS:
                values = group[metric].dropna()
                record[f"median_{metric}"] = (
                    float(values.median()) if not values.empty else np.nan
                )
                record[f"q25_{metric}"] = (
                    float(values.quantile(0.25)) if not values.empty else np.nan
                )
                record[f"q75_{metric}"] = (
                    float(values.quantile(0.75)) if not values.empty else np.nan
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


def _plot_figure_one(summary: pd.DataFrame, subclasses: list[str]):
    figure, axes = plt.subplots(
        len(subclasses),
        len(SLOTS),
        figsize=(19, max(3.0 * len(subclasses), 7)),
        squeeze=False,
        sharey=True,
    )
    for row_index, climate in enumerate(subclasses):
        for column_index, (slot, slot_label, color, marker) in enumerate(SLOTS):
            axis = axes[row_index, column_index]
            row = summary.loc[
                summary["climate"].eq(climate) & summary["slot"].eq(slot)
            ].iloc[0]
            median = row["median_gamma_suhi"]
            if pd.notna(median):
                axis.errorbar(
                    [0],
                    [median],
                    yerr=[
                        [median - row["q25_gamma_suhi"]],
                        [row["q75_gamma_suhi"] - median],
                    ],
                    color=color,
                    marker=marker,
                    markersize=6,
                    linewidth=1.2,
                    capsize=4,
                )
            axis.axhline(0, color="0.35", linewidth=0.7, alpha=0.7)
            axis.set_xlim(-0.5, 0.5)
            axis.set_xticks([0], [f"n={int(row['n_cities'])}"])
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(f"{climate} — {slot_label}")
            if column_index == 0:
                axis.set_ylabel("γSUHI (°C/decade)")
    figure.suptitle(
        "Summer-minus-winter SUHI trend by Köppen subclass\n"
        "City median and IQR; n = eligible cities",
        y=0.995,
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def _plot_figure_two(summary: pd.DataFrame, subclasses: list[str]):
    figure, axes = plt.subplots(
        len(subclasses),
        len(SLOTS),
        figsize=(19, max(3.0 * len(subclasses), 7)),
        squeeze=False,
        sharey=True,
    )
    positions = np.arange(len(GAMMA_METRICS), dtype=float)
    for row_index, climate in enumerate(subclasses):
        for column_index, (slot, slot_label, _, _) in enumerate(SLOTS):
            axis = axes[row_index, column_index]
            row = summary.loc[
                summary["climate"].eq(climate) & summary["slot"].eq(slot)
            ].iloc[0]
            for position, (metric, label, color) in zip(
                positions, GAMMA_METRICS
            ):
                median = row[f"median_{metric}"]
                if pd.isna(median):
                    continue
                axis.errorbar(
                    [position],
                    [median],
                    yerr=[
                        [median - row[f"q25_{metric}"]],
                        [row[f"q75_{metric}"] - median],
                    ],
                    color=color,
                    marker="o",
                    markersize=5,
                    linewidth=1.0,
                    capsize=4,
                    label=label,
                )
            axis.axhline(0, color="0.35", linewidth=0.7, alpha=0.7)
            axis.set_xlim(-0.5, len(GAMMA_METRICS) - 0.5)
            axis.set_xticks(positions, ["Urban", "Rural", "SUHI"])
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(f"{climate} — {slot_label}")
            if column_index == 0:
                axis.set_ylabel("Summer − winter γ (°C/decade)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        "Summer-minus-winter trend decomposition by Köppen subclass\n"
        "City-level γSUHI = γU − γR checked before summarization; "
        "subclass medians are shown independently",
        y=0.995,
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _plot_synchrony_scatter(
    frame: pd.DataFrame,
    category_column: str,
    category_label: str,
    major_climate_group: str,
    hemisphere: str,
):
    category_values = frame[category_column].astype("string").str.strip()
    categories = sorted(category_values.unique().tolist())
    color_map = plt.get_cmap("tab20")
    colors = color_map(np.linspace(0, 1, max(len(categories), 1)))
    category_colors = dict(zip(categories, colors))

    max_abs_value = max(
        float(frame["gamma_rural"].abs().max()),
        float(frame["gamma_urban"].abs().max()),
    )
    limit = max(max_abs_value * 1.05, 0.1)
    figure, axes = plt.subplots(
        1,
        len(SLOTS),
        figsize=(19, 5.8),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for column_index, (slot, slot_label, _, _) in enumerate(SLOTS):
        axis = axes[0, column_index]
        slot_frame = frame.loc[frame["slot"].eq(slot)]
        slot_categories = category_values.loc[slot_frame.index]
        for category in categories:
            category_rows = slot_frame.loc[
                slot_categories.eq(category).to_numpy()
            ]
            if category_rows.empty:
                continue
            axis.scatter(
                category_rows["gamma_rural"],
                category_rows["gamma_urban"],
                color=category_colors[category],
                s=15,
                alpha=0.65,
                linewidths=0,
                label=category if column_index == 0 else "_nolegend_",
            )
        axis.plot(
            [-limit, limit],
            [-limit, limit],
            color="0.2",
            linestyle="--",
            linewidth=1.1,
        )
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_title(f"{slot_label} (n={len(slot_frame):,})")
        axis.set_xlabel("γR (°C/decade)")
        if column_index == 0:
            axis.set_ylabel("γU (°C/decade)")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=category_colors[category],
            markeredgecolor="none",
            label=category,
        )
        for category in categories
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle="--",
            linewidth=1.1,
            label="y = x (γSUHI = 0)",
        )
    )
    figure.suptitle(
        f"City-level seasonal trend synchrony by {category_label}\n"
        f"Major climate {major_climate_group}; hemisphere: {hemisphere}; "
        "above/below y=x indicates positive/negative γSUHI",
        y=0.99,
        fontsize=13,
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 5),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.tight_layout(rect=(0, 0.14, 1, 0.9))
    return figure


def build_city_seasonal_gamma_subclimate_figures(
    config_path: Path,
) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = Path(config["paths"]["gamma_file"])
    if not source.is_file():
        raise FileNotFoundError(f"City seasonal gamma table not found: {source}")
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
        max_decomposition_error = _validate_gamma_trends(
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
            raise ValueError("No city/slot gamma rows remain for plotting.")
        print(
            f"[INFO] Major climate={config['major_climate_group']}; "
            f"subclasses={','.join(subclasses)}; "
            f"hemisphere={config['hemisphere']}; "
            f"cities={climate_counts['selected_cities']:,}; "
            f"rows={climate_counts['selected_rows']:,}.",
            flush=True,
        )

        summary = summarize_subclimate_gammas(selected, subclasses)
        summary_path = staging / "subclimate_seasonal_gamma_summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.8g")

        figure_one = _plot_figure_one(summary, subclasses)
        figure_one.suptitle(
            "Summer-minus-winter SUHI trend by Köppen subclass\n"
            f"Major climate {config['major_climate_group']}; "
            f"hemisphere: {config['hemisphere']}; "
            "city median and IQR; n = eligible cities",
            y=0.995,
            fontsize=14,
        )
        figure_one.tight_layout(rect=(0, 0, 1, 0.97))
        figure_one_path = staging / "figure_1_subclimate_gamma_suhi.png"
        figure_one.savefig(
            figure_one_path, dpi=config["dpi"], bbox_inches="tight"
        )
        plt.close(figure_one)

        figure_two = _plot_figure_two(summary, subclasses)
        figure_two.suptitle(
            "Summer-minus-winter trend decomposition by Köppen subclass\n"
            "City-level γSUHI = γU − γR checked before summarization; "
            "subclass medians are shown independently; "
            f"major climate {config['major_climate_group']}; "
            f"hemisphere: {config['hemisphere']}",
            y=0.995,
            fontsize=14,
        )
        figure_two.tight_layout(rect=(0, 0, 1, 0.94))
        figure_two_path = (
            staging / "figure_2_subclimate_gamma_decomposition.png"
        )
        figure_two.savefig(
            figure_two_path, dpi=config["dpi"], bbox_inches="tight"
        )
        plt.close(figure_two)

        scatter_subclimate = _plot_synchrony_scatter(
            selected,
            "climate",
            "Köppen subclass",
            config["major_climate_group"],
            config["hemisphere"],
        )
        scatter_subclimate_path = (
            staging / "figure_3_city_seasonal_synchrony_by_subclimate.png"
        )
        scatter_subclimate.savefig(
            scatter_subclimate_path,
            dpi=config["dpi"],
            bbox_inches="tight",
        )
        plt.close(scatter_subclimate)

        scatter_region = _plot_synchrony_scatter(
            selected,
            "GC_DEV_USR",
            "GHSL region",
            config["major_climate_group"],
            config["hemisphere"],
        )
        scatter_region_path = (
            staging / "figure_4_city_seasonal_synchrony_by_region.png"
        )
        scatter_region.savefig(
            scatter_region_path,
            dpi=config["dpi"],
            bbox_inches="tight",
        )
        plt.close(scatter_region)

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
            "summer_minus_winter_direction": {
                "north": "JJA - DJF",
                "south": "DJF - JJA",
            },
            "city_level_identity": "gamma_suhi = gamma_urban - gamma_rural",
            "max_absolute_city_level_decomposition_error": (
                max_decomposition_error
            ),
            "decomposition_tolerance": config["decomposition_tolerance"],
            "synchrony_scatter": {
                "x": "gamma_rural",
                "y": "gamma_urban",
                "reference_line": "y = x",
                "gamma_suhi_interpretation": "gamma_urban - gamma_rural",
                "color_groups": {
                    "subclimate": "climate",
                    "region": "GC_DEV_USR",
                },
            },
            "summary_note": (
                "Urban, rural, and SUHI subclass medians and IQRs are "
                "computed independently across cities and are not additive."
            ),
            "outputs": [
                figure_one_path.name,
                figure_two_path.name,
                scatter_subclimate_path.name,
                scatter_region_path.name,
                summary_path.name,
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
        print(
            f"[DONE] Subclimate gamma figures and summary published at "
            f"{output_dir}",
            flush=True,
        )
        return output_dir
    finally:
        connection.close()
        if not published and staging.exists():
            shutil.rmtree(staging)

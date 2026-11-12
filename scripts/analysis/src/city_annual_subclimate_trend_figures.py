"""Plot annual city trends by Köppen subclass within one major climate."""
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

from src.city_season_slot_trend_figures import (
    METRICS,
    SLOTS,
    _filter_trends_by_hemisphere,
)
from src.city_season_subclimate_trend_figures import _select_subclimates


REQUIRED_COLUMNS = {
    "ID_UC_G0",
    "climate",
    "centroid_lat",
    "hemisphere",
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
    config["_config_path"] = str(config_path)
    return config


def _validate_trends(
    frame: pd.DataFrame,
    tolerance: float,
) -> float:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Annual trend table is missing fields: {sorted(missing)}.")
    if frame.empty:
        raise ValueError("Annual trend table contains no city/slot rows.")
    if frame.duplicated(["ID_UC_G0", "slot"]).any():
        raise ValueError("Annual trend table has duplicate city/slot keys.")
    if frame["ID_UC_G0"].isna().any():
        raise ValueError("Annual trend table contains null city IDs.")
    valid_slots = {slot for slot, _, _, _ in SLOTS}
    if not set(frame["slot"].dropna()).issubset(valid_slots):
        raise ValueError("Annual trend table contains unsupported slots.")
    if frame["centroid_lat"].isna().any():
        raise ValueError("Annual trend table contains rows without centroid latitude.")

    metric_columns = ["beta_urban", "beta_rural", "beta_suhi"]
    values = frame[metric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("Annual trend coefficients must all be finite.")
    errors = (
        values["beta_suhi"]
        - (values["beta_urban"] - values["beta_rural"])
    ).abs()
    max_error = float(errors.max())
    if max_error > tolerance:
        raise ValueError(
            "City-level annual trend decomposition failed: "
            f"max |beta_suhi - (beta_urban - beta_rural)|={max_error:.12g}, "
            f"tolerance={tolerance:.12g}."
        )
    frame.loc[:, metric_columns] = values
    return max_error


def summarize_annual_subclimates(
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
                summary["climate"].eq(climate)
                & summary["slot"].eq(slot)
            ].iloc[0]
            median = row["median_beta_suhi"]
            if pd.notna(median):
                axis.errorbar(
                    [0],
                    [median],
                    yerr=[
                        [
                            median - row["q25_beta_suhi"],
                        ],
                        [
                            row["q75_beta_suhi"] - median,
                        ],
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
                axis.set_ylabel("βSUHI (°C/decade)")
    figure.suptitle(
        "Annual SUHI trend by Köppen subclass\n"
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
    positions = np.arange(len(METRICS), dtype=float)
    for row_index, climate in enumerate(subclasses):
        for column_index, (slot, slot_label, _, _) in enumerate(SLOTS):
            axis = axes[row_index, column_index]
            row = summary.loc[
                summary["climate"].eq(climate)
                & summary["slot"].eq(slot)
            ].iloc[0]
            for position, (metric, label, color) in zip(positions, METRICS):
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
            axis.set_xlim(-0.5, len(METRICS) - 0.5)
            axis.set_xticks(positions, ["Urban", "Rural", "SUHI"])
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(f"{climate} — {slot_label}")
            if column_index == 0:
                axis.set_ylabel("Slope (°C/decade)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        "Annual urban–rural trend decomposition by Köppen subclass\n"
        "City-level βSUHI = βU − βR checked before summarization; "
        "subclass medians are shown independently",
        y=0.995,
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def build_city_annual_subclimate_trend_figures(
    config_path: Path,
) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = Path(config["paths"]["trend_file"])
    if not source.is_file():
        raise FileNotFoundError(f"City annual trend table not found: {source}")
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
        max_decomposition_error = _validate_trends(
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
            raise ValueError("No annual trends remain for plotting.")
        print(
            f"[INFO] Major climate={config['major_climate_group']}; "
            f"subclasses={','.join(subclasses)}; "
            f"hemisphere={config['hemisphere']}; "
            f"cities={climate_counts['selected_cities']:,}; "
            f"rows={climate_counts['selected_rows']:,}.",
            flush=True,
        )

        summary = summarize_annual_subclimates(selected, subclasses)
        summary_path = staging / "annual_subclimate_trend_summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.8g")

        figure_one = _plot_figure_one(summary, subclasses)
        figure_one.suptitle(
            "Annual SUHI trend by Köppen subclass\n"
            f"Major climate {config['major_climate_group']}; "
            f"hemisphere: {config['hemisphere']}; "
            "city median and IQR; n = eligible cities",
            y=0.995,
            fontsize=14,
        )
        figure_one.tight_layout(rect=(0, 0, 1, 0.97))
        figure_one_path = staging / "figure_1_annual_subclimate_suhi_trends.png"
        figure_one.savefig(
            figure_one_path, dpi=config["dpi"], bbox_inches="tight"
        )
        plt.close(figure_one)

        figure_two = _plot_figure_two(summary, subclasses)
        figure_two.suptitle(
            "Annual urban–rural trend decomposition by Köppen subclass\n"
            "City-level βSUHI = βU − βR checked before summarization; "
            "subclass medians are shown independently; "
            f"major climate {config['major_climate_group']}; "
            f"hemisphere: {config['hemisphere']}",
            y=0.995,
            fontsize=14,
        )
        figure_two.tight_layout(rect=(0, 0, 1, 0.94))
        figure_two_path = (
            staging / "figure_2_annual_subclimate_urban_rural_decomposition.png"
        )
        figure_two.savefig(
            figure_two_path, dpi=config["dpi"], bbox_inches="tight"
        )
        plt.close(figure_two)

        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trend_file": source.name,
            "trend_sha256": _sha256(source),
            "config_file": config_path.name,
            "config_sha256": _sha256(config_path),
            "major_climate_group": config["major_climate_group"],
            "subclimate_classes": subclasses,
            "hemisphere_filter": hemisphere_filter,
            "climate_group_counts": climate_counts,
            "city_level_identity": "beta_suhi = beta_urban - beta_rural",
            "max_absolute_city_level_decomposition_error": (
                max_decomposition_error
            ),
            "decomposition_tolerance": config["decomposition_tolerance"],
            "summary_note": (
                "Urban, rural, and SUHI subclass medians and IQRs are "
                "computed independently across cities and are not additive."
            ),
            "outputs": [
                figure_one_path.name,
                figure_two_path.name,
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
        print(f"[DONE] Annual subclimate figures published at {output_dir}", flush=True)
        return output_dir
    finally:
        connection.close()
        if not published and staging.exists():
            shutil.rmtree(staging)

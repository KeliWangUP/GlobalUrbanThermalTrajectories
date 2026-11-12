"""Plot seasonal city trends by Köppen subclass within one major climate."""
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
import matplotlib.pyplot as plt
import pandas as pd

from src.city_season_slot_trend_figures import (
    SEASONS,
    SLOTS,
    _filter_trends_by_hemisphere,
    _plot_figure_one,
    _plot_figure_two,
    _validate_trends,
    summarize_trends,
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
            "Config 'decomposition_tolerance' must be a non-negative number."
        )
    config["_config_path"] = str(config_path)
    return config


def _select_subclimates(
    frame: pd.DataFrame,
    major_climate_group: str,
    requested_classes: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    climate_codes = frame["climate"].astype("string").str.strip()
    normalized_codes = climate_codes.str.upper()
    climate_groups = normalized_codes.str[0]
    major_mask = climate_groups.eq(major_climate_group).fillna(False)
    major_frame = frame.loc[major_mask].copy()
    major_frame["climate"] = climate_codes.loc[major_mask]
    available = sorted(major_frame["climate"].dropna().unique().tolist())
    if not available:
        raise ValueError(
            f"No trend rows found for Köppen major group "
            f"'{major_climate_group}'."
        )

    available_by_normalized = {item.upper(): item for item in available}
    requested_normalized = [item.upper() for item in requested_classes]
    unavailable = sorted(set(requested_normalized) - set(available_by_normalized))
    if unavailable:
        raise ValueError(
            "Requested Köppen subclasses have no trend rows for the selected "
            f"hemisphere: {unavailable}."
        )
    selected_classes = (
        [available_by_normalized[item] for item in requested_normalized]
        if requested_normalized
        else available
    )
    selected_normalized = {item.upper() for item in selected_classes}
    selected = major_frame.loc[
        major_frame["climate"].astype("string").str.upper().isin(
            selected_normalized
        )
    ].copy()
    counts = {
        "source_rows": len(frame),
        "source_cities": int(frame["ID_UC_G0"].nunique()),
        "major_group_rows": len(major_frame),
        "major_group_cities": int(major_frame["ID_UC_G0"].nunique()),
        "selected_rows": len(selected),
        "selected_cities": int(selected["ID_UC_G0"].nunique()),
        "excluded_other_major_rows": len(frame) - len(major_frame),
        "excluded_other_major_cities": int(
            frame.loc[~major_mask, "ID_UC_G0"].nunique()
        ),
    }
    return selected, selected_classes, counts


def summarize_subclimate_trends(
    frame: pd.DataFrame,
    subclasses: list[str],
) -> pd.DataFrame:
    grouped = frame.copy()
    grouped["climate_group"] = grouped["climate"]
    return summarize_trends(grouped, subclasses)


def _write_summary_csv(summary: pd.DataFrame, path: Path) -> None:
    summary.to_csv(path, index=False, float_format="%.8g")


def build_city_season_subclimate_trend_figures(
    config_path: Path,
) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = Path(config["paths"]["trend_file"])
    if not source.is_file():
        raise FileNotFoundError(f"City-season trend table not found: {source}")
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
        max_decomposition_error = _validate_trends(frame, config)
        frame, hemisphere_filter = _filter_trends_by_hemisphere(
            frame, config["hemisphere"]
        )
        selected, subclasses, climate_counts = _select_subclimates(
            frame,
            config["major_climate_group"],
            config["subclimate_classes"],
        )
        if selected.empty:
            raise ValueError(
                "No city/season/slot trend rows remain for plotting."
            )

        print(
            f"[INFO] Major climate={config['major_climate_group']}; "
            f"subclasses={','.join(subclasses)}; "
            f"hemisphere={config['hemisphere']}; "
            f"cities={climate_counts['selected_cities']:,}; "
            f"rows={climate_counts['selected_rows']:,}.",
            flush=True,
        )
        summary = summarize_subclimate_trends(selected, subclasses)
        summary_path = staging / "subclimate_season_slot_trend_summary.csv"
        _write_summary_csv(summary, summary_path)

        figure_one = _plot_figure_one(summary, subclasses)
        figure_one.suptitle(
            "Seasonal SUHI trends by Köppen subclass\n"
            f"Major climate {config['major_climate_group']}; "
            f"hemisphere: {config['hemisphere']}; median with IQR; "
            "n = eligible cities",
            y=0.995,
            fontsize=14,
        )
        figure_one.tight_layout(rect=(0, 0, 1, 0.97))
        figure_one_path = (
            staging / "figure_1_subclimate_seasonal_suhi_trends.png"
        )
        figure_one.savefig(
            figure_one_path, dpi=config["dpi"], bbox_inches="tight"
        )
        plt.close(figure_one)

        figure_two = _plot_figure_two(summary, subclasses)
        figure_two.suptitle(
            "Urban–Rural seasonal trend decomposition by Köppen subclass\n"
            "City-level βSUHI = βU − βR checked before summarization; "
            "subclass medians are shown independently; "
            f"major climate {config['major_climate_group']}; "
            f"hemisphere: {config['hemisphere']}",
            y=0.995,
            fontsize=14,
        )
        figure_two.tight_layout(rect=(0, 0, 1, 0.94))
        figure_two_path = (
            staging / "figure_2_subclimate_urban_rural_decomposition.png"
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
            "seasons": list(SEASONS),
            "slots": [slot for slot, _, _, _ in SLOTS],
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
        print(
            f"[DONE] Subclimate figures and summary published at {output_dir}",
            flush=True,
        )
        return output_dir
    finally:
        connection.close()
        if not published and staging.exists():
            shutil.rmtree(staging)

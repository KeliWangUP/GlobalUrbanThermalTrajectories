#!/usr/bin/env python3
"""Create QC-completeness tables and monthly trend-distribution plots from Gold."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

try:
    import duckdb
except ImportError as exc:
    raise SystemExit(
        "DuckDB is required. Install project dependencies with "
        "python -m pip install -r requirements.txt."
    ) from exc


MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MAJOR_ZONES = ("A", "B", "C", "D")
FOCUS_CLASSES = ("Af", "Am", "Aw")
SLOTS = (
    ("td", "terra_day", "Terra", "Day"),
    ("tn", "terra_night", "Terra", "Night"),
    ("ad", "aqua_day", "Aqua", "Day"),
    ("an", "aqua_night", "Aqua", "Night"),
)
SLOT_ORDER = tuple(slot[1] for slot in SLOTS)
SENSOR_COLORS = {"Terra": "#D55E00", "Aqua": "#0072B2"}
METRICS = {
    "suhi": ("suhi_trend_decade", r"$\beta_{SUHI}$"),
    "urban": ("urban_trend_decade", r"$\beta_U$"),
    "rural": ("rural_trend_decade", r"$\beta_R$"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise ValueError("Diagnostics config must contain a 'paths' object.")
    paths = config["paths"]
    for key in ("gold_dir", "output_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Diagnostics config 'paths.{key}' must be a path.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths[key] = path.resolve().as_posix()
    expected = config.get("expected_city_count")
    if type(expected) is not int or expected < 1:
        raise ValueError("Diagnostics config 'expected_city_count' must be positive.")
    plot = config.get("plot")
    if not isinstance(plot, dict):
        raise ValueError("Diagnostics config must contain a 'plot' object.")
    dpi = plot.get("dpi")
    if type(dpi) is not int or dpi < 1:
        raise ValueError("Diagnostics config 'plot.dpi' must be a positive integer.")
    config["_config_path"] = str(config_path)
    return config


def _wide_schema_columns() -> tuple[list[str], list[str]]:
    qc_columns = []
    year_columns = []
    for code, _, _, _ in SLOTS:
        for month in range(1, 13):
            suffix = f"{code}_m{month:02d}"
            qc_columns.append(f"qc_{suffix}")
            year_columns.append(f"n_years_{suffix}")
    return qc_columns, year_columns


def _validate_wide(wide: pd.DataFrame, expected_city_count: int) -> None:
    required = {
        "ID_UC_G0",
        "koppen_geiger_class",
        "climate_major",
    }
    qc_columns, year_columns = _wide_schema_columns()
    missing = (required | set(qc_columns) | set(year_columns)) - set(wide.columns)
    if missing:
        raise ValueError(f"Gold wide table is missing fields: {sorted(missing)}.")
    if len(wide) != expected_city_count:
        raise ValueError(
            f"Gold wide table has {len(wide)} rows; expected "
            f"{expected_city_count} cities."
        )
    if wide["ID_UC_G0"].isna().any() or wide["ID_UC_G0"].duplicated().any():
        raise ValueError("Gold wide table must contain one non-null row per city.")


def _validate_long(long: pd.DataFrame) -> None:
    required = {
        "ID_UC_G0",
        "koppen_geiger_class",
        "climate_major",
        "month",
        "slot",
        "qc_pass",
        *[column for column, _ in METRICS.values()],
    }
    missing = required - set(long.columns)
    if missing:
        raise ValueError(f"Gold long table is missing fields: {sorted(missing)}.")
    if not long.empty:
        if not long["month"].between(1, 12).all():
            raise ValueError("Gold long table contains invalid month values.")
        if not set(long["slot"].dropna()).issubset(SLOT_ORDER):
            raise ValueError("Gold long table contains unknown sensor slots.")
        if long.duplicated(["ID_UC_G0", "month", "slot"]).any():
            raise ValueError("Gold long table has duplicate city/month/slot rows.")


def summarize_qc_completeness(wide: pd.DataFrame) -> pd.DataFrame:
    """Count passing, failing, and zero-year fits for each climate/month/slot."""
    climate_major = wide["climate_major"].astype("string").fillna("NoData")
    climate_class = wide["koppen_geiger_class"].astype("string").fillna("NoData")
    major_groups = {
        zone: wide.loc[climate_major == zone]
        for zone in (*MAJOR_ZONES, "E", "NoData")
    }
    class_zones = sorted(
        set(climate_class.tolist()) | set(FOCUS_CLASSES) | {"NoData"}
    )
    class_groups = {
        zone: wide.loc[climate_class == zone]
        for zone in class_zones
    }

    rows = []
    for level, groups in (("major", major_groups), ("class", class_groups)):
        for zone, group in groups.items():
            total = len(group)
            for month in range(1, 13):
                for code, slot, sensor, overpass in SLOTS:
                    suffix = f"{code}_m{month:02d}"
                    qc_col = f"qc_{suffix}"
                    years_col = f"n_years_{suffix}"
                    passed = group[qc_col].fillna(False).astype(bool)
                    valid_years = pd.to_numeric(
                        group[years_col], errors="raise"
                    ).fillna(0)
                    pass_count = int(passed.sum())
                    rows.append({
                        "climate_level": level,
                        "climate_zone": zone,
                        "month": month,
                        "slot": slot,
                        "sensor": sensor,
                        "overpass": overpass,
                        "n_cities": total,
                        "n_cities_with_fit": int(valid_years.gt(0).sum()),
                        "n_cities_pass": pass_count,
                        "n_cities_fail": total - pass_count,
                        "n_cities_zero_valid_years": int(valid_years.eq(0).sum()),
                        "pass_fraction": pass_count / total if total else np.nan,
                    })
    return pd.DataFrame(rows)


def _plot_qc_heatmap(
    summary: pd.DataFrame,
    climate_level: str,
    zones: tuple[str, ...],
    path: Path,
    title: str,
    dpi: int,
) -> None:
    n_zones = len(zones)
    ncols = 2 if n_zones > 2 else n_zones
    nrows = (n_zones + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.8 * ncols, 4.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, zone in zip(axes.flat, zones):
        subset = summary.loc[
            (summary["climate_level"] == climate_level)
            & (summary["climate_zone"] == zone)
        ]
        rates = subset.pivot(
            index="slot", columns="month", values="pass_fraction"
        ).reindex(index=SLOT_ORDER, columns=range(1, 13))
        image = axis.imshow(
            rates.to_numpy(dtype=float),
            vmin=0,
            vmax=1,
            cmap="RdYlGn",
            aspect="auto",
        )
        axis.set_title(zone)
        axis.set_xticks(range(12), MONTH_LABELS)
        axis.set_yticks(
            range(len(SLOT_ORDER)),
            ("Terra day", "Terra night", "Aqua day", "Aqua night"),
        )
        axis.set_xlabel("Calendar month")
        for row_index, slot in enumerate(SLOT_ORDER):
            for month in range(1, 13):
                cell = subset.loc[
                    (subset["slot"] == slot) & (subset["month"] == month)
                ]
                if cell.empty or int(cell.iloc[0]["n_cities"]) == 0:
                    label = "—\n0/0"
                else:
                    item = cell.iloc[0]
                    label = (
                        f"{item['pass_fraction']:.0%}\n"
                        f"{int(item['n_cities_pass'])}/{int(item['n_cities'])}"
                    )
                axis.text(
                    month - 1,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    for axis in axes.flat[n_zones:]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(
            image,
            ax=list(axes.flat[:n_zones]),
            label="Trend QC pass fraction",
            shrink=0.85,
        )
    figure.suptitle(title)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_qc_heatmaps(
    summary: pd.DataFrame,
    output_dir: Path,
    label: str,
    dpi: int,
) -> list[Path]:
    major_path = output_dir / f"qc_pass_rate_major_{label}.png"
    focus_path = output_dir / f"qc_pass_rate_Af_Am_Aw_{label}.png"
    _plot_qc_heatmap(
        summary,
        "major",
        (*MAJOR_ZONES,),
        major_path,
        f"Trend QC pass fraction by Köppen major climate zone ({label})",
        dpi,
    )
    _plot_qc_heatmap(
        summary,
        "class",
        FOCUS_CLASSES,
        focus_path,
        f"Monthly trend QC by tropical climate class ({label})",
        dpi,
    )
    return [major_path, focus_path]


def plot_trend_distributions(
    long: pd.DataFrame,
    output_dir: Path,
    label: str,
    dpi: int,
) -> list[Path]:
    """Plot month-by-month city distributions; no cold/warm seasons are imposed."""
    passing = long.loc[
        long["qc_pass"].fillna(False).astype(bool)
        & long["climate_major"].isin(MAJOR_ZONES)
    ]
    outputs = []
    for zone in MAJOR_ZONES:
        zone_data = passing.loc[passing["climate_major"] == zone]
        for metric, (column, beta_label) in METRICS.items():
            figure, axes = plt.subplots(
                2, 1, figsize=(14, 8), sharex=True, constrained_layout=True
            )
            for axis, overpass in zip(axes, ("Day", "Night")):
                has_boxes = False
                for month in range(1, 13):
                    for _, slot, sensor, slot_overpass in SLOTS:
                        if slot_overpass != overpass:
                            continue
                        values = pd.to_numeric(
                            zone_data.loc[
                                (zone_data["month"] == month)
                                & (zone_data["slot"] == slot),
                                column,
                            ],
                            errors="raise",
                        ).dropna().to_numpy()
                        if values.size == 0:
                            continue
                        offset = -0.18 if sensor == "Terra" else 0.18
                        boxes = axis.boxplot(
                            values,
                            positions=[month + offset],
                            widths=0.30,
                            patch_artist=True,
                            showfliers=False,
                            manage_ticks=False,
                        )
                        boxes["boxes"][0].set_facecolor(SENSOR_COLORS[sensor])
                        boxes["boxes"][0].set_alpha(0.65)
                        has_boxes = True
                axis.axhline(0, color="black", linewidth=0.8, alpha=0.7)
                axis.set_ylabel(f"{beta_label}\nper decade")
                axis.set_title(f"{overpass} overpass")
                axis.set_xlim(0.5, 12.5)
                axis.set_xticks(range(1, 13), MONTH_LABELS)
                axis.grid(axis="y", alpha=0.25)
                if not has_boxes:
                    axis.text(
                        0.5,
                        0.5,
                        "No QC-passing trends",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                    )
            axes[-1].set_xlabel("Calendar month")
            figure.legend(
                handles=[
                    Patch(
                        facecolor=SENSOR_COLORS[sensor],
                        edgecolor="black",
                        alpha=0.65,
                        label=sensor,
                    )
                    for sensor in ("Terra", "Aqua")
                ],
                loc="upper center",
                ncol=2,
            )
            figure.suptitle(
                f"Climate {zone}: {beta_label} monthly distributions ({label})"
            )
            path = output_dir / f"beta_{metric}_climate_{zone}_{label}.png"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            outputs.append(path)
    return outputs


def _read_gold_table(connection, path: Path) -> pd.DataFrame:
    return connection.execute(
        "SELECT * FROM read_parquet(?)",
        [path.as_posix()],
    ).df()


def run_diagnostics(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    paths = config["paths"]
    gold_dir = Path(paths["gold_dir"])
    gold_manifest_path = gold_dir / "manifest.json"
    if not gold_manifest_path.is_file():
        raise FileNotFoundError(f"Gold manifest not found: {gold_manifest_path}")
    gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8"))
    windows = gold_manifest.get("trend_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Gold manifest has no trend_windows list.")

    output_dir = Path(paths["output_dir"])
    if output_dir.exists():
        raise FileExistsError(
            f"Diagnostics destination already exists; refusing to overwrite: "
            f"{output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}_staging_",
            dir=output_dir.parent,
        )
    )
    print(f"[INFO] Diagnostics staging directory: {staging}", flush=True)

    connection = duckdb.connect(":memory:")
    outputs: list[dict[str, Any]] = []
    try:
        for window in windows:
            if (
                not isinstance(window, list)
                or len(window) != 2
                or any(type(year) is not int for year in window)
            ):
                raise ValueError(f"Invalid trend window in Gold manifest: {window}")
            start, end = window
            label = f"{start}_{end}"
            wide_path = gold_dir / f"city_month_trend_wide_{label}.parquet"
            long_path = gold_dir / f"city_month_slot_trends_{label}.parquet"
            if not wide_path.is_file() or not long_path.is_file():
                raise FileNotFoundError(
                    f"Gold trend files are incomplete for window {label}."
                )

            wide = _read_gold_table(connection, wide_path)
            long = _read_gold_table(connection, long_path)
            _validate_wide(wide, config["expected_city_count"])
            _validate_long(long)

            summary = summarize_qc_completeness(wide)
            summary_path = staging / f"qc_completeness_{label}.csv"
            summary.to_csv(summary_path, index=False, float_format="%.6f")
            focus_path = staging / f"qc_Af_Am_Aw_{label}.csv"
            summary.loc[
                (summary["climate_level"] == "class")
                & summary["climate_zone"].isin(FOCUS_CLASSES)
            ].to_csv(focus_path, index=False, float_format="%.6f")
            outputs.extend(
                [
                    {"file": summary_path.name, "rows": len(summary)},
                    {
                        "file": focus_path.name,
                        "rows": int(
                            (
                                (summary["climate_level"] == "class")
                                & summary["climate_zone"].isin(FOCUS_CLASSES)
                            ).sum()
                        ),
                    },
                ]
            )
            for figure_path in plot_qc_heatmaps(
                summary, staging, label, config["plot"]["dpi"]
            ):
                outputs.append({"file": figure_path.name})
            for figure_path in plot_trend_distributions(
                long, staging, label, config["plot"]["dpi"]
            ):
                outputs.append({"file": figure_path.name})
            print(
                f"[WINDOW {label}] cities={len(wide):,}; "
                f"QC summary rows={len(summary):,}; "
                f"QC-passing long rows={int(long['qc_pass'].fillna(False).sum()):,}.",
                flush=True,
            )
    finally:
        connection.close()

    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gold_manifest_sha256": _sha256(gold_manifest_path),
        "diagnostics_config_sha256": _sha256(config_path),
        "expected_city_count": config["expected_city_count"],
        "qc_completeness_denominator": (
            "All cities in the climate zone, including zero-valid-year fits."
        ),
        "wet_season_note": (
            "No wet-season months are labeled: the current inputs have no "
            "monthly precipitation series. Af/Am/Aw are reported by calendar "
            "month so wet-season alignment can be assessed without imposing "
            "a global month definition."
        ),
        "outputs": outputs,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Diagnostics destination appeared during the build: {output_dir}"
        )
    staging.rename(output_dir)
    print(f"[DONE] Gold diagnostics published at {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name(
            "city_month_gold_diagnostics_config.json"
        ),
        help="Path to the Gold diagnostics config JSON.",
    )
    args = parser.parse_args()
    run_diagnostics(args.config)


if __name__ == "__main__":
    main()

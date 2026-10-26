import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from src.eda_climate_month_trends import (
    TREND_METRICS,
    summarize_climate_month_slopes
)
from src.diurnal_eda import MONTH_NAMES


LOGGER = logging.getLogger("diurnal_eda")
CLIMATE_COLUMN = "koppen_geiger_macro_group"
METRIC_LABELS = {
    "day": "βD",
    "night": "βN",
    "asymmetry": "βA"
}


def _metric_limit(
    summaries: list[pd.DataFrame],
    metric: str
) -> float:
    columns = (
        f"median_beta_{metric}",
        f"ci_lower_beta_{metric}",
        f"ci_upper_beta_{metric}"
    )
    values = [
        abs(float(value))
        for summary in summaries
        for column in columns
        for value in summary[column].dropna()
        if np.isfinite(value)
    ]
    if not values:
        return 0.05
    return max(max(values) * 1.05, 0.05)


def _create_heatmap(
    summary: pd.DataFrame,
    sensor: str,
    metric: str,
    climate_groups: tuple[str, ...],
    lower_limit: float,
    confidence_level: float
):
    value_column = f"median_beta_{metric}"
    lower_column = f"ci_lower_beta_{metric}"
    upper_column = f"ci_upper_beta_{metric}"
    matrix = (
        summary.pivot(
            index=CLIMATE_COLUMN,
            columns="month",
            values=value_column
        )
        .reindex(index=climate_groups, columns=range(1, 13))
    )
    counts = (
        summary.pivot(
            index=CLIMATE_COLUMN,
            columns="month",
            values="n_cities"
        )
        .reindex(index=climate_groups, columns=range(1, 13))
    )
    ci_lower = (
        summary.pivot(
            index=CLIMATE_COLUMN,
            columns="month",
            values=lower_column
        )
        .reindex(index=climate_groups, columns=range(1, 13))
    )
    ci_upper = (
        summary.pivot(
            index=CLIMATE_COLUMN,
            columns="month",
            values=upper_column
        )
        .reindex(index=climate_groups, columns=range(1, 13))
    )

    figure, axis = plt.subplots(figsize=(15, 5.7))
    image = axis.imshow(
        matrix.to_numpy(dtype="float64"),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-lower_limit, vcenter=0, vmax=lower_limit),
        aspect="auto",
        interpolation="nearest"
    )
    axis.set_xticks(range(12), [name[:3] for name in MONTH_NAMES])
    axis.set_yticks(range(len(climate_groups)), climate_groups)
    axis.set_xlabel("Local-season month")
    axis.set_ylabel("Köppen-Geiger macroclimate group")
    axis.set_title(
        f"{sensor.title()} {METRIC_LABELS[metric]} trend by climate group "
        "and month"
    )

    values = matrix.to_numpy(dtype="float64")
    city_counts = counts.to_numpy(dtype="float64")
    lows = ci_lower.to_numpy(dtype="float64")
    highs = ci_upper.to_numpy(dtype="float64")
    for row in range(len(climate_groups)):
        for column in range(12):
            value = values[row, column]
            count = city_counts[row, column]
            if not np.isfinite(value) or count == 0:
                label = "—"
                text_color = "0.35"
            else:
                excludes_zero = (
                    count > 1
                    and (
                        lows[row, column] > 0
                        or highs[row, column] < 0
                    )
                )
                marker = "*" if excludes_zero else ""
                label = f"{value:.2f}{marker}\nn={int(count)}"
                text_color = "white" if abs(value) > lower_limit * 0.55 else "black"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color
            )

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("City-level slope median (°C per decade)")
    figure.text(
        0.99,
        0.01,
        f"* {confidence_level:.0%} bootstrap CI excludes zero (n > 1)",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35"
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def create_climate_month_trend_heatmaps(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    equatorial_band_degrees: float,
    minimum_years: int,
    climate_groups: tuple[str, ...],
    bootstrap_replicates: int,
    confidence_level: float,
    random_seed: int,
    dpi: int
) -> list[Path]:
    """Plot climate × local-month heatmaps for city-level D/N/A slopes."""
    summaries = {}
    for sensor_index, sensor in enumerate(("terra", "aqua")):
        summaries[sensor] = summarize_climate_month_slopes(
            panel=panel,
            sensor=sensor,
            start_year=start_year,
            end_year=end_year,
            equatorial_band_degrees=equatorial_band_degrees,
            minimum_years=minimum_years,
            climate_groups=climate_groups,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            random_seed=(random_seed + sensor_index) % (2**32)
        )

    output_dir = output_dir / "climate_month_trend_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sensor, summary in summaries.items():
        long_summary = summary.melt(
            id_vars=[CLIMATE_COLUMN, "month", "n_cities"],
            value_vars=[
                f"{bound}_beta_{metric}"
                for metric in TREND_METRICS
                for bound in ("median", "ci_lower", "ci_upper")
            ],
            var_name="statistic",
            value_name="value"
        )
        long_summary.insert(0, "sensor", sensor)
        long_summary["confidence_level"] = confidence_level
        long_summary["bootstrap_replicates"] = bootstrap_replicates
        records.append(long_summary)

    statistics_path = output_dir / (
        f"climate_month_D_N_A_trend_statistics_{start_year}-{end_year}.csv"
    )
    pd.concat(records, ignore_index=True).to_csv(
        statistics_path,
        index=False
    )
    LOGGER.info("Climate-month trend statistics: %s", statistics_path)

    outputs = []
    for metric in TREND_METRICS:
        limit = _metric_limit(list(summaries.values()), metric)
        for sensor, summary in summaries.items():
            figure = _create_heatmap(
                summary=summary,
                sensor=sensor,
                metric=metric,
                climate_groups=climate_groups,
                lower_limit=limit,
                confidence_level=confidence_level
            )
            output = output_dir / (
                f"{sensor}_beta_{metric}_climate_by_month_"
                f"{start_year}-{end_year}.png"
            )
            figure.savefig(output, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            LOGGER.info("Figure: %s", output)
            outputs.append(output)
    return outputs

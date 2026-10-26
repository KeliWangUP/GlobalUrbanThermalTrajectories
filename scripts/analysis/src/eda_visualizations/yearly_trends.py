import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.diurnal_eda import LATITUDE_GROUP_ORDER, MONTH_NAMES
from src.eda_trends import TREND_METRICS, annual_sensor_trend_summary


LOGGER = logging.getLogger("diurnal_eda")
METRIC_LABELS = {
    "day": "Median D",
    "night": "Median N",
    "asymmetry": "Median A (D - N)",
    "mean": "Median M (intensity)"
}
METRIC_COLORS = {
    "day": "#d55e00",
    "night": "#0072b2",
    "asymmetry": "#009e73",
    "mean": "#cc79a7"
}


def _symmetric_limits(summaries: list[pd.DataFrame]) -> tuple[float, float]:
    values = [
        float(value)
        for summary in summaries
        for metric in TREND_METRICS
        for column in (
            f"median_{metric}",
            f"ci_lower_{metric}",
            f"ci_upper_{metric}"
        )
        for value in summary[column].dropna()
        if np.isfinite(value)
    ]
    if not values:
        raise ValueError("No QC-passing annual D/N/A/M values are available.")
    limit = max(max(abs(value) for value in values) * 1.05, 0.25)
    return -limit, limit


def _year_ticks(start_year: int, end_year: int) -> list[int]:
    ticks = list(range(start_year, end_year + 1, 5))
    if not ticks or ticks[-1] != end_year:
        ticks.append(end_year)
    return ticks


def _create_trend_grid(
    summary: pd.DataFrame,
    sensor: str,
    latitude_group: str,
    start_year: int,
    end_year: int,
    lower: float,
    upper: float,
    confidence_level: float
):
    years = list(range(start_year, end_year + 1))
    figure, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(16, 10),
        sharex=True,
        sharey=True
    )

    group_summary = summary.loc[
        summary["latitude_group"].eq(latitude_group)
    ]
    for month, axis in enumerate(axes.flat, start=1):
        monthly = (
            group_summary.loc[group_summary["month"].eq(month)]
            .set_index("year")
            .reindex(years)
        )
        city_counts = monthly["n_cities"].fillna(0).to_numpy(dtype="int64")
        if not city_counts.any():
            axis.text(
                0.5,
                0.5,
                "No QC-passing cities",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=9
            )
        else:
            for metric in TREND_METRICS:
                median = monthly[f"median_{metric}"].to_numpy(
                    dtype="float64"
                )
                ci_lower = monthly[f"ci_lower_{metric}"].to_numpy(
                    dtype="float64"
                )
                ci_upper = monthly[f"ci_upper_{metric}"].to_numpy(
                    dtype="float64"
                )
                color = METRIC_COLORS[metric]
                axis.plot(
                    years,
                    median,
                    color=color,
                    marker="o",
                    markersize=3,
                    linewidth=1.4
                )
                axis.fill_between(
                    years,
                    ci_lower,
                    ci_upper,
                    where=np.isfinite(ci_lower) & np.isfinite(ci_upper),
                    color=color,
                    alpha=0.13,
                    linewidth=0
                )

            present_counts = city_counts[city_counts > 0]
            axis.text(
                0.97,
                0.04,
                f"n={present_counts.min()}-{present_counts.max()}",
                ha="right",
                va="bottom",
                transform=axis.transAxes,
                fontsize=8,
                color="0.35"
            )

        axis.axhline(0, color="0.5", linewidth=0.6, zorder=0)
        axis.set_xlim(start_year - 0.5, end_year + 0.5)
        axis.set_ylim(lower, upper)
        axis.set_xticks(_year_ticks(start_year, end_year))
        axis.set_title(MONTH_NAMES[month - 1])
        axis.grid(alpha=0.2, linewidth=0.5)

        if city_counts.any():
            present_counts = city_counts[city_counts > 0]
            LOGGER.info(
                "[YEARLY TREND] %s %s %s: cities/year min=%s median=%s "
                "max=%s; years without data=%s",
                sensor.upper(),
                latitude_group,
                MONTH_NAMES[month - 1],
                int(present_counts.min()),
                int(np.median(present_counts)),
                int(present_counts.max()),
                int((city_counts == 0).sum())
            )
        else:
            LOGGER.info(
                "[YEARLY TREND] %s %s %s: no QC-passing city observations",
                sensor.upper(),
                latitude_group,
                MONTH_NAMES[month - 1]
            )

    handles = [
        Line2D(
            [0],
            [0],
            color=METRIC_COLORS[metric],
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=METRIC_LABELS[metric]
        )
        for metric in TREND_METRICS
    ]
    confidence_percent = confidence_level * 100
    figure.suptitle(
        f"{sensor.title()} {latitude_group}: annual median D, N, A, and M "
        f"with {confidence_percent:g}% bootstrap CI",
        fontsize=14
    )
    figure.supxlabel("Year")
    figure.supylabel("Median SUHI / asymmetry (°C)")
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False
    )
    figure.tight_layout(rect=(0.04, 0.04, 0.97, 0.88))
    return figure


def create_yearly_trend_figures(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    equatorial_band_degrees: float,
    bootstrap_replicates: int,
    confidence_level: float,
    random_seed: int,
    dpi: int
) -> list[Path]:
    """Plot annual median D/N/A/M and city-bootstrap confidence bands."""
    summaries = []
    for sensor_index, sensor in enumerate(("terra", "aqua")):
        summaries.append(
            annual_sensor_trend_summary(
                panel=panel,
                sensor=sensor,
                start_year=start_year,
                end_year=end_year,
                equatorial_band_degrees=equatorial_band_degrees,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                random_seed=(random_seed + sensor_index) % (2**32)
            )
        )
    lower, upper = _symmetric_limits(summaries)

    outputs = []
    for sensor, summary in zip(("terra", "aqua"), summaries):
        for latitude_group in LATITUDE_GROUP_ORDER:
            figure = _create_trend_grid(
                summary=summary,
                sensor=sensor,
                latitude_group=latitude_group,
                start_year=start_year,
                end_year=end_year,
                lower=lower,
                upper=upper,
                confidence_level=confidence_level
            )
            group_slug = latitude_group.lower().replace(" ", "_")
            group_dir = output_dir / "yearly_trends" / sensor / group_slug
            group_dir.mkdir(parents=True, exist_ok=True)
            confidence_percent = int(round(confidence_level * 100))
            output = group_dir / (
                f"{sensor}_{group_slug}_median_D_N_A_M_"
                f"bootstrap_ci{confidence_percent}_"
                f"{start_year}-{end_year}.png"
            )
            figure.savefig(output, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            LOGGER.info("Figure: %s", output)
            outputs.append(output)
    return outputs

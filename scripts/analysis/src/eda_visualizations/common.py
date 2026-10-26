from math import isfinite

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from src.diurnal_eda import MONTH_NAMES


SENSOR_COLORS = {"terra": "#d55e00", "aqua": "#0072b2"}


def symmetric_axis_limits(
    frames: list[pd.DataFrame]
) -> tuple[float, float]:
    values = [
        float(value)
        for frame in frames
        for column in ("day_suhi", "night_suhi")
        if column in frame.columns
        for value in frame[column].dropna()
        if isfinite(float(value))
    ]
    if not values:
        raise ValueError("No finite QC-passing day/night SUHI values to plot.")
    limit = max(max(abs(value) for value in values) * 1.05, 0.25)
    return -limit, limit


def configure_diurnal_axis(axis, lower: float, upper: float) -> None:
    axis.set_axisbelow(True)
    axis.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=0.9,
        color="0.35"
    )
    axis.spines["left"].set_position("zero")
    axis.spines["bottom"].set_position("zero")
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.xaxis.set_ticks_position("bottom")
    axis.yaxis.set_ticks_position("left")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2, linewidth=0.5)


def scatter_city_points(
    axis,
    frame: pd.DataFrame,
    sensor: str,
    size: int,
    alpha: float
) -> None:
    axis.scatter(
        frame["day_suhi"],
        frame["night_suhi"],
        s=size,
        alpha=alpha,
        color=SENSOR_COLORS[sensor],
        edgecolors="none",
        zorder=3
    )


def legend_handles(sensor: str) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=SENSOR_COLORS[sensor],
            label="City"
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="0.35",
            label="Day = night"
        )
    ]


def create_month_grid(
    frame: pd.DataFrame,
    sensor: str,
    lower: float,
    upper: float,
    title: str,
    x_label: str
):
    figure, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(16, 11),
        sharex=True,
        sharey=True
    )
    for month, axis in enumerate(axes.flat, start=1):
        month_frame = frame.loc[frame["month"] == month]
        if month_frame.empty:
            axis.text(
                0.5,
                0.5,
                "No QC-passing cities",
                ha="center",
                va="center",
                transform=axis.transAxes
            )
        else:
            scatter_city_points(
                axis,
                month_frame,
                sensor,
                size=18,
                alpha=0.6
            )
            axis.text(
                0.04,
                0.96,
                f"n = {month_frame['ID_UC_G0'].nunique():,} cities",
                ha="left",
                va="top",
                transform=axis.transAxes,
                fontsize=9
            )
        configure_diurnal_axis(axis, lower, upper)
        axis.set_title(MONTH_NAMES[month - 1])

    figure.suptitle(title, fontsize=15)
    figure.supxlabel(x_label)
    figure.supylabel("Nighttime SUHI (°C)")
    figure.legend(
        handles=legend_handles(sensor),
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=False
    )
    figure.tight_layout(rect=(0.04, 0.04, 0.96, 0.92))
    return figure


def create_year_grid(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    sensor: str,
    month: int,
    start_year: int,
    end_year: int,
    lower: float,
    upper: float,
    title: str,
    x_label: str
):
    years = list(range(start_year, end_year + 1))
    columns = 5
    row_count = (len(years) + columns - 1) // columns
    figure, axes = plt.subplots(
        nrows=row_count,
        ncols=columns,
        figsize=(columns * 3.4, row_count * 3.2),
        sharex=True,
        sharey=True,
        squeeze=False
    )
    for panel_index, year in enumerate(years):
        axis = axes.flat[panel_index]
        year_frame = rows.loc[
            rows["year"].eq(year)
            & rows["month"].eq(month)
        ]
        year_summary = summary.loc[(year, month)]
        if year_frame.empty:
            message = (
                "No records"
                if year_summary["total_city_months"] == 0
                else "No QC-passing cities"
            )
            axis.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=8
            )
        else:
            scatter_city_points(
                axis,
                year_frame,
                sensor,
                size=14,
                alpha=0.55
            )
            axis.text(
                0.04,
                0.96,
                f"n = {year_frame['ID_UC_G0'].nunique():,}",
                ha="left",
                va="top",
                transform=axis.transAxes,
                fontsize=8
            )
        axis.set_title(str(year), fontsize=10)
        configure_diurnal_axis(axis, lower, upper)

    for axis in axes.flat[len(years):]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=15)
    figure.supxlabel(x_label)
    figure.supylabel("Nighttime SUHI (°C)")
    figure.legend(
        handles=legend_handles(sensor),
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=False
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.97, 0.94))
    return figure

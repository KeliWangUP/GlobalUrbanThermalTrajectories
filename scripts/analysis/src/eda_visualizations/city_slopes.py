import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.diurnal_eda import LATITUDE_GROUP_ORDER, MONTH_NAMES
from src.eda_city_slopes import calculate_city_month_slopes


LOGGER = logging.getLogger("diurnal_eda")
LATITUDE_GROUP_COLORS = {
    "Northern Hemisphere": "#0072b2",
    "Southern Hemisphere": "#d55e00",
    "Equatorial Belt": "#009e73"
}


def _symmetric_slope_limits(
    slopes_by_sensor: list[pd.DataFrame]
) -> tuple[float, float]:
    values = [
        float(value)
        for slopes in slopes_by_sensor
        for column in ("beta_day_per_decade", "beta_night_per_decade")
        for value in slopes[column]
        if np.isfinite(value)
    ]
    if not values:
        raise ValueError("No city D/N slopes are available to plot.")
    limit = max(max(abs(value) for value in values) * 1.05, 0.05)
    return -limit, limit


def _create_city_slope_grid(
    slopes: pd.DataFrame,
    coverage: pd.DataFrame,
    sensor: str,
    latitude_group: str,
    start_year: int,
    end_year: int,
    minimum_years: int,
    lower: float,
    upper: float
):
    figure, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(15, 11),
        sharex=True,
        sharey=True
    )
    color = LATITUDE_GROUP_COLORS[latitude_group]

    for month, axis in enumerate(axes.flat, start=1):
        month_slopes = slopes.loc[
            slopes["latitude_group"].eq(latitude_group)
            & slopes["month"].eq(month)
        ]
        month_coverage = coverage.loc[
            coverage["latitude_group"].eq(latitude_group)
            & coverage["month"].eq(month)
        ].iloc[0]
        if month_slopes.empty:
            axis.text(
                0.5,
                0.5,
                f"No cities with ≥{minimum_years} valid years",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=8
            )
        else:
            axis.scatter(
                month_slopes["beta_day_per_decade"],
                month_slopes["beta_night_per_decade"],
                s=18,
                alpha=0.55,
                color=color,
                edgecolors="none",
                rasterized=True
            )
            axis.text(
                0.04,
                0.96,
                f"n = {len(month_slopes):,} cities",
                ha="left",
                va="top",
                transform=axis.transAxes,
                fontsize=8
            )
        axis.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=0.8,
            color="0.4",
            zorder=1
        )
        axis.axhline(0, color="0.65", linewidth=0.55, zorder=0)
        axis.axvline(0, color="0.65", linewidth=0.55, zorder=0)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18, linewidth=0.45)
        axis.set_title(MONTH_NAMES[month - 1])

        LOGGER.info(
            "[CITY SLOPE] %s %s %s: plotted=%s, QC cities=%s, "
            "below minimum years=%s (minimum=%s)",
            sensor.upper(),
            latitude_group,
            MONTH_NAMES[month - 1],
            int(month_coverage["n_cities_plotted"]),
            int(month_coverage["n_cities_with_qc_data"]),
            int(month_coverage["n_cities_below_minimum"]),
            minimum_years
        )

    group_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        color=color,
        label=latitude_group
    )
    diagonal_handle = Line2D(
        [0],
        [0],
        linestyle="--",
        color="0.4",
        label="βD = βN"
    )
    figure.suptitle(
        f"{sensor.title()} city-specific D/N SUHI trends — {latitude_group} "
        f"({start_year}–{end_year}; ≥{minimum_years} valid years)",
        fontsize=14
    )
    figure.supxlabel("βD (°C per decade)")
    figure.supylabel("βN (°C per decade)")
    figure.legend(
        handles=[group_handle, diagonal_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False
    )
    figure.tight_layout(rect=(0.04, 0.04, 0.97, 0.88))
    return figure


def create_city_slope_figures(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    equatorial_band_degrees: float,
    minimum_years: int,
    dpi: int
) -> list[Path]:
    """Plot city-specific annual D/N slopes by calendar month and latitude."""
    summaries = []
    for sensor in ("terra", "aqua"):
        slopes, coverage = calculate_city_month_slopes(
            panel=panel,
            sensor=sensor,
            start_year=start_year,
            end_year=end_year,
            equatorial_band_degrees=equatorial_band_degrees,
            minimum_years=minimum_years
        )
        summaries.append((sensor, slopes, coverage))
    lower, upper = _symmetric_slope_limits(
        [slopes for _, slopes, _ in summaries]
    )

    outputs = []
    for sensor, slopes, coverage in summaries:
        for latitude_group in LATITUDE_GROUP_ORDER:
            figure = _create_city_slope_grid(
                slopes=slopes,
                coverage=coverage,
                sensor=sensor,
                latitude_group=latitude_group,
                start_year=start_year,
                end_year=end_year,
                minimum_years=minimum_years,
                lower=lower,
                upper=upper
            )
            group_slug = latitude_group.lower().replace(" ", "_")
            group_dir = output_dir / "city_slope_scatter" / sensor / group_slug
            group_dir.mkdir(parents=True, exist_ok=True)
            output = group_dir / (
                f"{sensor}_{group_slug}_city_beta_D_N_by_month_"
                f"{start_year}-{end_year}_min{minimum_years}years.png"
            )
            figure.savefig(output, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            LOGGER.info("Figure: %s", output)
            outputs.append(output)
    return outputs

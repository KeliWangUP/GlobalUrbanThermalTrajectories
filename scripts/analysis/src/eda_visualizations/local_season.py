import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.diurnal_eda import (
    CITY_COLUMN,
    LATITUDE_GROUP_ORDER,
    MONTH_COLUMN,
    MONTH_NAMES,
    YEAR_COLUMN,
    add_local_season_coordinates,
    aggregate_sensor_local_city_months,
    log_sensor_summary,
    log_year_month_summary,
    prepare_sensor_year_month_rows
)
from src.eda_visualizations.common import (
    create_month_grid,
    create_year_grid,
    symmetric_axis_limits
)


LOGGER = logging.getLogger("diurnal_eda")


def _log_local_month_counts(
    sensor: str,
    latitude_group: str,
    frame: pd.DataFrame,
    start_year: int,
    end_year: int
) -> None:
    LOGGER.info(
        "[LOCAL MONTH] %s %s city counts by local month, source years %s-%s",
        sensor.upper(),
        latitude_group,
        start_year,
        end_year
    )
    for month in range(1, 13):
        month_frame = frame.loc[frame[MONTH_COLUMN].eq(month)]
        LOGGER.info(
            "%s: %s cities",
            MONTH_NAMES[month - 1],
            f"{month_frame[CITY_COLUMN].nunique():,}"
        )


def create_local_season_month_figures(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    aggregation: str,
    dpi: int,
    equatorial_band_degrees: float
) -> list[Path]:
    """Create separate local-season figures for each latitude group."""
    data = {}
    for sensor in ("terra", "aqua"):
        city_months, summary = aggregate_sensor_local_city_months(
            panel,
            sensor,
            start_year,
            end_year,
            aggregation,
            equatorial_band_degrees
        )
        data[sensor] = (city_months, summary)
        log_sensor_summary(
            f"{sensor} local season",
            summary,
            start_year,
            end_year
        )

    lower, upper = symmetric_axis_limits(
        [city_months for city_months, _ in data.values()]
    )
    outputs = []
    for sensor, (city_months, _) in data.items():
        for latitude_group in LATITUDE_GROUP_ORDER:
            group = city_months.loc[
                city_months["latitude_group"].eq(latitude_group)
            ].copy()
            _log_local_month_counts(
                sensor,
                latitude_group,
                group,
                start_year,
                end_year
            )
            figure = create_month_grid(
                frame=group,
                sensor=sensor,
                lower=lower,
                upper=upper,
                title=(
                    f"{sensor.title()} {latitude_group}: "
                    f"daytime vs nighttime "
                    f"SUHI by local-season month ({aggregation}, "
                    f"{start_year}-{end_year})"
                ),
                x_label=(
                    "Daytime SUHI (°C); month aligned to Northern Hemisphere"
                )
            )
            group_slug = latitude_group.lower().replace(" ", "_")
            group_dir = output_dir / "local_season" / group_slug
            group_dir.mkdir(parents=True, exist_ok=True)
            output = group_dir / (
                f"{sensor}_{group_slug}_local_month_suhi_{aggregation}_"
                f"{start_year}-{end_year}.png"
            )
            figure.savefig(output, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            LOGGER.info("Figure: %s", output)
            outputs.append(output)
    return outputs


def _empty_year_summary(
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [range(start_year, end_year + 1), range(1, 13)],
        names=[YEAR_COLUMN, MONTH_COLUMN]
    )
    summary = pd.DataFrame(index=index)
    for column in (
        "total_city_months",
        "total_cities",
        "qc_pass_city_months",
        "qc_pass_cities",
        "plotted_cities"
    ):
        summary[column] = 0
    summary["qc_pass_rate_pct"] = 0.0
    summary["city_coverage_pct"] = 0.0
    return summary


def create_local_season_year_figures(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    dpi: int,
    equatorial_band_degrees: float
) -> list[Path]:
    """Create separate monthly year-facet figures for three latitude groups."""
    aligned_start = start_year + 1
    if aligned_start > end_year:
        raise ValueError(
            "At least two source years are required to create complete "
            "latitude-aligned year panels."
        )

    aligned = add_local_season_coordinates(
        panel,
        equatorial_band_degrees
    )
    aligned[YEAR_COLUMN] = aligned["local_year"]
    aligned[MONTH_COLUMN] = aligned["local_month"]
    aligned = aligned.loc[
        aligned[YEAR_COLUMN].between(aligned_start, end_year)
    ].copy()
    if aligned.empty:
        raise ValueError(
            f"No complete local-season years found for "
            f"{aligned_start}-{end_year}."
        )

    groups = {}
    all_frames = []
    for sensor in ("terra", "aqua"):
        groups[sensor] = {}
        for latitude_group in LATITUDE_GROUP_ORDER:
            subset = aligned.loc[
                aligned["latitude_group"].eq(latitude_group)
            ].copy()
            if subset.empty:
                rows = pd.DataFrame(
                    columns=[
                        CITY_COLUMN,
                        YEAR_COLUMN,
                        MONTH_COLUMN,
                        "day_suhi",
                        "night_suhi",
                        "latitude_group"
                    ]
                )
                summary = _empty_year_summary(aligned_start, end_year)
            else:
                rows, summary = prepare_sensor_year_month_rows(
                    subset,
                    sensor,
                    aligned_start,
                    end_year
                )
            groups[sensor][latitude_group] = (rows, summary)
            all_frames.append(rows)

    lower, upper = symmetric_axis_limits(all_frames)
    outputs = []
    for sensor in ("terra", "aqua"):
        for latitude_group in LATITUDE_GROUP_ORDER:
            rows, summary = groups[sensor][latitude_group]
            group_slug = latitude_group.lower().replace(" ", "_")
            for month in range(1, 13):
                log_year_month_summary(
                    sensor,
                    month,
                    summary,
                    aligned_start,
                    end_year,
                    group_name=latitude_group
                )
                figure = create_year_grid(
                    rows=rows,
                    summary=summary,
                    sensor=sensor,
                    month=month,
                    start_year=aligned_start,
                    end_year=end_year,
                    lower=lower,
                    upper=upper,
                    title=(
                        f"{sensor.title()} {latitude_group} "
                        f"{MONTH_NAMES[month - 1]}: daytime vs nighttime "
                        "SUHI by aligned year"
                    ),
                    x_label=(
                        "Daytime SUHI (°C); month aligned to Northern "
                        "Hemisphere"
                    )
                )
                group_dir = (
                    output_dir / "local_season" / "by_year" / group_slug
                )
                group_dir.mkdir(parents=True, exist_ok=True)
                output = group_dir / (
                    f"{sensor}_{group_slug}_"
                    f"{MONTH_NAMES[month - 1].lower()}_"
                    f"by_aligned_year_{aligned_start}-{end_year}.png"
                )
                figure.savefig(output, dpi=dpi, bbox_inches="tight")
                plt.close(figure)
                LOGGER.info("Figure: %s", output)
                outputs.append(output)
    return outputs

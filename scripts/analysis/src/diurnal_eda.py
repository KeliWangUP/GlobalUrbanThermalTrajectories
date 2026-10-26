import logging
from math import isfinite
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


CITY_COLUMN = "ID_UC_G0"
YEAR_COLUMN = "year"
MONTH_COLUMN = "month"
KEY_COLUMNS = [CITY_COLUMN, YEAR_COLUMN, MONTH_COLUMN]
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
)
SENSOR_COLUMNS = {
    "terra": {
        "day": "suhi_terra_day_mean",
        "night": "suhi_terra_night_mean",
        "qc": "qc_pass_A_T"
    },
    "aqua": {
        "day": "suhi_aqua_day_mean",
        "night": "suhi_aqua_night_mean",
        "qc": "qc_pass_A_A"
    }
}
SENSOR_COLORS = {"terra": "#d55e00", "aqua": "#0072b2"}
HEMISPHERE_COLORS = {
    "Northern Hemisphere": "#0072b2",
    "Southern Hemisphere": "#d55e00",
    "Equator": "#009e73"
}
LOGGER = logging.getLogger("diurnal_eda")


def _prepare_sensor_month_rows(
    panel: pd.DataFrame,
    sensor: str,
    start_year: int,
    end_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a sensor's QC-paired source and valid city-month rows."""
    if sensor not in SENSOR_COLUMNS:
        raise ValueError(f"Unsupported sensor: {sensor}")
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")

    sensor_columns = SENSOR_COLUMNS[sensor]
    required_columns = [
        *KEY_COLUMNS,
        sensor_columns["day"],
        sensor_columns["night"],
        sensor_columns["qc"]
    ]
    if "hemisphere" in panel.columns:
        required_columns.append("hemisphere")
    missing = [column for column in required_columns if column not in panel]
    if missing:
        raise ValueError(f"Monthly SUHI panel is missing columns: {missing}")
    if panel.duplicated(KEY_COLUMNS).any():
        raise ValueError("Monthly SUHI panel contains duplicate city/month keys.")

    in_period = panel.loc[
        panel[YEAR_COLUMN].between(start_year, end_year),
        required_columns
    ].copy()
    if in_period.empty:
        raise ValueError(
            f"No monthly SUHI records found for {start_year}-{end_year}."
        )
    if not in_period[MONTH_COLUMN].between(1, 12).all():
        raise ValueError("Month values in the monthly SUHI panel must be 1..12.")
    if not pd.api.types.is_bool_dtype(in_period[sensor_columns["qc"]].dtype):
        raise TypeError(
            f"{sensor_columns['qc']} must contain boolean QC flags."
        )

    paired_pass = (
        in_period[sensor_columns["qc"]].fillna(False)
        & in_period[sensor_columns["day"]].notna()
        & in_period[sensor_columns["night"]].notna()
    )
    valid = in_period.loc[paired_pass].copy()
    return in_period, valid


def aggregate_sensor_city_months(
    panel: pd.DataFrame,
    sensor: str,
    start_year: int,
    end_year: int,
    aggregation: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate paired day/night SUHI values by city and calendar month."""
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be 'mean' or 'median'.")
    in_period, valid = _prepare_sensor_month_rows(
        panel,
        sensor,
        start_year,
        end_year
    )
    sensor_columns = SENSOR_COLUMNS[sensor]

    valid_summary = (
        valid
        .groupby(MONTH_COLUMN)
        .agg(
            qc_pass_city_year_months=(CITY_COLUMN, "size"),
            qc_pass_cities=(CITY_COLUMN, "nunique")
        )
        .reindex(range(1, 13), fill_value=0)
    )
    source_summary = (
        in_period
        .groupby(MONTH_COLUMN)
        .agg(
            total_city_year_months=(CITY_COLUMN, "size"),
            total_cities=(CITY_COLUMN, "nunique")
        )
        .reindex(range(1, 13), fill_value=0)
    )
    summary = source_summary.join(valid_summary)
    summary["qc_pass_rate_pct"] = (
        summary["qc_pass_city_year_months"]
        .div(summary["total_city_year_months"].where(
            summary["total_city_year_months"] > 0
        ))
        .mul(100)
        .fillna(0)
    )
    summary["city_coverage_pct"] = (
        summary["qc_pass_cities"]
        .div(summary["total_cities"].where(summary["total_cities"] > 0))
        .mul(100)
        .fillna(0)
    )

    city_month = (
        valid
        .groupby([CITY_COLUMN, MONTH_COLUMN], sort=True)
        .agg(
            day_suhi=(sensor_columns["day"], aggregation),
            night_suhi=(sensor_columns["night"], aggregation),
            n_valid_years=(YEAR_COLUMN, "nunique")
        )
        .reset_index()
    )
    if "hemisphere" in valid.columns:
        city_hemispheres = (
            valid.groupby(CITY_COLUMN)["hemisphere"]
            .first()
            .rename("hemisphere")
            .reset_index()
        )
        city_month = city_month.merge(
            city_hemispheres,
            on=CITY_COLUMN,
            validate="many_to_one"
        )
    plotted_cities = (
        city_month
        .groupby(MONTH_COLUMN)[CITY_COLUMN]
        .nunique()
        .reindex(range(1, 13), fill_value=0)
    )
    summary["plotted_cities"] = plotted_cities.astype("int64")
    summary.index.name = MONTH_COLUMN
    return city_month, summary


def prepare_sensor_year_month_rows(
    panel: pd.DataFrame,
    sensor: str,
    start_year: int,
    end_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired QC-passing city observations and year/month counts."""
    in_period, valid = _prepare_sensor_month_rows(
        panel,
        sensor,
        start_year,
        end_year
    )
    index = pd.MultiIndex.from_product(
        [range(start_year, end_year + 1), range(1, 13)],
        names=[YEAR_COLUMN, MONTH_COLUMN]
    )
    source_summary = (
        in_period
        .groupby([YEAR_COLUMN, MONTH_COLUMN])
        .agg(
            total_city_months=(CITY_COLUMN, "size"),
            total_cities=(CITY_COLUMN, "nunique")
        )
        .reindex(index, fill_value=0)
    )
    valid_summary = (
        valid
        .groupby([YEAR_COLUMN, MONTH_COLUMN])
        .agg(
            qc_pass_city_months=(CITY_COLUMN, "size"),
            qc_pass_cities=(CITY_COLUMN, "nunique")
        )
        .reindex(index, fill_value=0)
    )
    summary = source_summary.join(valid_summary)
    count_columns = [
        "total_city_months",
        "total_cities",
        "qc_pass_city_months",
        "qc_pass_cities"
    ]
    summary[count_columns] = summary[count_columns].astype("int64")
    summary["qc_pass_rate_pct"] = (
        summary["qc_pass_city_months"]
        .div(summary["total_city_months"].where(
            summary["total_city_months"] > 0
        ))
        .mul(100)
        .fillna(0)
    )
    summary["city_coverage_pct"] = (
        summary["qc_pass_cities"]
        .div(summary["total_cities"].where(summary["total_cities"] > 0))
        .mul(100)
        .fillna(0)
    )
    summary["plotted_cities"] = summary["qc_pass_cities"]

    sensor_columns = SENSOR_COLUMNS[sensor]
    plotted_rows = valid[
        [
            CITY_COLUMN,
            YEAR_COLUMN,
            MONTH_COLUMN,
            sensor_columns["day"],
            sensor_columns["night"]
        ]
    ].rename(columns={
        sensor_columns["day"]: "day_suhi",
        sensor_columns["night"]: "night_suhi"
    })
    if "hemisphere" in valid.columns:
        plotted_rows["hemisphere"] = valid["hemisphere"]
    return plotted_rows, summary


def add_local_season_coordinates(panel: pd.DataFrame) -> pd.DataFrame:
    """Shift southern observations by six months and derive aligned year."""
    required = {*KEY_COLUMNS, "centroid_lat"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
        "Local-season alignment requires columns: "
        f"{sorted(missing)}"
        )
    if panel.duplicated(KEY_COLUMNS).any():
        raise ValueError("Monthly SUHI panel contains duplicate city/month keys.")
    if not panel[MONTH_COLUMN].between(1, 12).all():
        raise ValueError("Month values in the monthly SUHI panel must be 1..12.")

    aligned = panel.copy()
    centroid_latitude = pd.to_numeric(
        aligned["centroid_lat"],
        errors="raise"
    )
    if centroid_latitude.isna().any() or not centroid_latitude.between(
        -90,
        90
    ).all():
        raise ValueError("centroid_lat values must be finite and in [-90, 90].")

    aligned["hemisphere"] = "Equator"
    aligned.loc[
        centroid_latitude.gt(0),
        "hemisphere"
    ] = "Northern Hemisphere"
    aligned.loc[
        centroid_latitude.lt(0),
        "hemisphere"
    ] = "Southern Hemisphere"

    month_index = (
        aligned[MONTH_COLUMN] - 1
        + centroid_latitude.lt(0).astype("int64") * 6
    )
    aligned["local_month"] = month_index.mod(12) + 1
    aligned["local_year"] = (
        aligned[YEAR_COLUMN] + month_index.floordiv(12)
    )
    return aligned


def aggregate_sensor_local_city_months(
    panel: pd.DataFrame,
    sensor: str,
    start_year: int,
    end_year: int,
    aggregation: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate sensor data by hemisphere-aligned local month."""
    aligned = add_local_season_coordinates(panel)
    local_panel = aligned.loc[
        aligned[YEAR_COLUMN].between(start_year, end_year)
    ].copy()
    local_panel[MONTH_COLUMN] = local_panel["local_month"]
    return aggregate_sensor_city_months(
        local_panel,
        sensor,
        start_year,
        end_year,
        aggregation
    )


def log_sensor_summary(
    sensor: str,
    summary: pd.DataFrame,
    start_year: int,
    end_year: int
) -> None:
    """Log QC coverage and plotted-city counts for each calendar month."""
    LOGGER.info(
        f"[EDA] {sensor.upper()} paired day/night QC, "
        f"{start_year}-{end_year}"
    )
    LOGGER.info(
        "month  passed_city_year_months/total  pass_rate  "
        "cities_with_pass/total  plotted_cities"
    )
    for month, row in summary.iterrows():
        LOGGER.info(
            f"{month:02d}     "
            f"{int(row['qc_pass_city_year_months']):,}/"
            f"{int(row['total_city_year_months']):,}  "
            f"{row['qc_pass_rate_pct']:.1f}%      "
            f"{int(row['qc_pass_cities']):,}/"
            f"{int(row['total_cities']):,}  "
            f"{int(row['plotted_cities']):,}"
        )


def _symmetric_axis_limits(values: list[float]) -> tuple[float, float]:
    limit = max(max(abs(value) for value in values) * 1.05, 0.25)
    return -limit, limit


def _configure_diurnal_axis(axis, lower: float, upper: float) -> None:
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


def _scatter_city_points(
    axis,
    frame: pd.DataFrame,
    sensor: str,
    size: int,
    alpha: float
) -> bool:
    """Plot city points, colored by hemisphere when that field is present."""
    if "hemisphere" not in frame.columns:
        axis.scatter(
            frame["day_suhi"],
            frame["night_suhi"],
            s=size,
            alpha=alpha,
            color=SENSOR_COLORS[sensor],
            edgecolors="none",
            zorder=3
        )
        return False

    unknown = set(frame["hemisphere"].dropna()) - set(HEMISPHERE_COLORS)
    if unknown or frame["hemisphere"].isna().any():
        raise ValueError(f"Unexpected hemisphere labels: {sorted(unknown)}")
    for hemisphere, color in HEMISPHERE_COLORS.items():
        hemisphere_frame = frame.loc[frame["hemisphere"] == hemisphere]
        if hemisphere_frame.empty:
            continue
        axis.scatter(
            hemisphere_frame["day_suhi"],
            hemisphere_frame["night_suhi"],
            s=size,
            alpha=alpha,
            color=color,
            edgecolors="none",
            zorder=3
        )
    return True


def _city_legend_handles(sensor: str, by_hemisphere: bool) -> list[Line2D]:
    if by_hemisphere:
        return [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=color,
                label=hemisphere
            )
            for hemisphere, color in HEMISPHERE_COLORS.items()
        ]
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=SENSOR_COLORS[sensor],
            label="City"
        )
    ]


def create_diurnal_sensor_figures(
    city_months: dict[str, pd.DataFrame],
    output_dir: Path,
    start_year: int,
    end_year: int,
    aggregation: str,
    dpi: int,
    local_season: bool = False
) -> list[Path]:
    """Create one 12-panel day/night SUHI scatter figure per sensor."""
    values = [
        float(value)
        for sensor_frame in city_months.values()
        for column in ("day_suhi", "night_suhi")
        for value in sensor_frame[column].dropna()
        if isfinite(float(value))
    ]
    if not values:
        raise ValueError(
            "No QC-passing Terra or Aqua day/night SUHI values are available "
            f"for {start_year}-{end_year}."
        )

    lower, upper = _symmetric_axis_limits(values)
    plot_directory = (
        output_dir / "local_season"
        if local_season
        else output_dir
    )
    plot_directory.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for sensor in ("terra", "aqua"):
        frame = city_months[sensor]
        figure, axes = plt.subplots(
            nrows=3,
            ncols=4,
            figsize=(16, 11),
            sharex=True,
            sharey=True
        )
        for month, axis in enumerate(axes.flat, start=1):
            month_frame = frame.loc[frame[MONTH_COLUMN] == month]
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
                _scatter_city_points(
                    axis,
                    month_frame,
                    sensor,
                    size=18,
                    alpha=0.6
                )
                axis.text(
                    0.04,
                    0.96,
                    f"n = {len(month_frame):,} cities",
                    ha="left",
                    va="top",
                    transform=axis.transAxes,
                    fontsize=9
                )
            _configure_diurnal_axis(axis, lower, upper)
            axis.set_title(MONTH_NAMES[month - 1])

        figure.suptitle(
            f"{sensor.title()} daytime vs nighttime SUHI"
            f"{' by local-season month' if local_season else ''} "
            f"({aggregation} by city-month, {start_year}-{end_year})",
            fontsize=15
        )
        figure.supxlabel(
            "Daytime SUHI (°C)"
            if not local_season
            else "Daytime SUHI (°C); month aligned to Northern Hemisphere"
        )
        figure.supylabel("Nighttime SUHI (°C)")
        figure.legend(
            handles=[
                *_city_legend_handles(
                    sensor,
                    by_hemisphere=local_season
                ),
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color="0.35",
                    label="Day = night"
                )
            ],
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            frameon=False
        )
        figure.tight_layout(rect=(0.04, 0.04, 0.96, 0.92))

        output_path = plot_directory / (
            f"{sensor}_{'local_season_' if local_season else ''}"
            f"day_vs_night_suhi_{aggregation}_"
            f"{start_year}-{end_year}.png"
        )
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def log_year_month_summary(
    sensor: str,
    month: int,
    summary: pd.DataFrame,
    start_year: int,
    end_year: int
) -> None:
    """Log yearly QC counts for one sensor and calendar month."""
    LOGGER.info(
        f"[EDA YEAR] {sensor.upper()} {MONTH_NAMES[month - 1]} "
        f"({start_year}-{end_year})"
    )
    LOGGER.info("year  passed_city_months/total  pass_rate  cities_with_pass")
    for year in range(start_year, end_year + 1):
        row = summary.loc[(year, month)]
        LOGGER.info(
            f"{year}  "
            f"{int(row['qc_pass_city_months']):,}/"
            f"{int(row['total_city_months']):,}  "
            f"{row['qc_pass_rate_pct']:.1f}%  "
            f"{int(row['qc_pass_cities']):,}"
        )


def _plot_month_by_year(
    sensor: str,
    month: int,
    sensor_rows: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    lower: float,
    upper: float,
    dpi: int,
    local_season: bool = False
) -> Path:
    years = list(range(start_year, end_year + 1))
    columns = 5
    rows = (len(years) + columns - 1) // columns
    figure, axes = plt.subplots(
        nrows=rows,
        ncols=columns,
        figsize=(columns * 3.4, rows * 3.2),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    for panel_index, year in enumerate(years):
        axis = axes.flat[panel_index]
        year_frame = sensor_rows.loc[
            sensor_rows[YEAR_COLUMN].eq(year)
            & sensor_rows[MONTH_COLUMN].eq(month)
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
            _scatter_city_points(
                axis,
                year_frame,
                sensor,
                size=14,
                alpha=0.55
            )
            axis.text(
                0.04,
                0.96,
                f"n = {year_frame[CITY_COLUMN].nunique():,}",
                ha="left",
                va="top",
                transform=axis.transAxes,
                fontsize=8
            )
        axis.set_title(str(year), fontsize=10)
        _configure_diurnal_axis(axis, lower, upper)

    for axis in axes.flat[len(years):]:
        axis.set_visible(False)

    figure.suptitle(
        f"{sensor.title()} {MONTH_NAMES[month - 1]}"
        f"{' local-season month' if local_season else ''}: "
        f"daytime vs nighttime SUHI by "
        f"{'aligned year' if local_season else 'year'}",
        fontsize=15
    )
    figure.supxlabel(
        "Daytime SUHI (°C)"
        if not local_season
        else "Daytime SUHI (°C); month aligned to Northern Hemisphere"
    )
    figure.supylabel("Nighttime SUHI (°C)")
    figure.legend(
        handles=[
            *_city_legend_handles(
                sensor,
                by_hemisphere="hemisphere" in sensor_rows.columns
            ),
            Line2D(
                [0],
                [0],
                linestyle="--",
                color="0.35",
                label="Day = night"
            )
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=False
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.97, 0.94))

    relative_dir = (
        Path("local_season") / "by_year"
        if local_season
        else Path("by_year")
    )
    sensor_dir = output_dir / relative_dir / sensor
    sensor_dir.mkdir(parents=True, exist_ok=True)
    output_path = sensor_dir / (
        f"{sensor}_{MONTH_NAMES[month - 1].lower()}_"
        f"day_vs_night_suhi_{'by_aligned_year' if local_season else 'by_year'}_"
        f"{start_year}-{end_year}.png"
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_year_panel_figures(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    dpi: int,
    local_season: bool = False
) -> list[Path]:
    """Create one 12-monthly-figure group per sensor, faceted by year."""
    sensor_rows = {}
    sensor_summaries = {}
    for sensor in ("terra", "aqua"):
        sensor_rows[sensor], sensor_summaries[sensor] = (
            prepare_sensor_year_month_rows(
                panel,
                sensor,
                start_year,
                end_year
            )
        )

    values = [
        float(value)
        for rows in sensor_rows.values()
        for column in ("day_suhi", "night_suhi")
        for value in rows[column].dropna()
        if isfinite(float(value))
    ]
    if not values:
        raise ValueError(
            "No QC-passing Terra or Aqua day/night SUHI values are available "
            f"for {start_year}-{end_year}."
        )
    lower, upper = _symmetric_axis_limits(values)

    output_paths = []
    for sensor in ("terra", "aqua"):
        summary = sensor_summaries[sensor]
        for month in range(1, 13):
            log_year_month_summary(
                sensor,
                month,
                summary,
                start_year,
                end_year
            )
            output_path = _plot_month_by_year(
                sensor=sensor,
                month=month,
                sensor_rows=sensor_rows[sensor],
                summary=summary,
                output_dir=output_dir,
                start_year=start_year,
                end_year=end_year,
                lower=lower,
                upper=upper,
                dpi=dpi,
                local_season=local_season
            )
            output_paths.append(output_path)
    return output_paths


def create_local_season_figures(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    aggregation: str,
    dpi: int
) -> list[Path]:
    """Create monthly day/night figures after aligning southern seasons."""
    aligned = add_local_season_coordinates(panel)
    source_period = aligned.loc[
        aligned[YEAR_COLUMN].between(start_year, end_year)
    ].copy()
    source_period[MONTH_COLUMN] = source_period["local_month"]

    city_months = {}
    for sensor in ("terra", "aqua"):
        city_months[sensor], summary = aggregate_sensor_city_months(
            source_period,
            sensor,
            start_year,
            end_year,
            aggregation
        )
        log_sensor_summary(
            f"{sensor} local season",
            summary,
            start_year,
            end_year
        )
    return create_diurnal_sensor_figures(
        city_months,
        output_dir,
        start_year,
        end_year,
        aggregation,
        dpi,
        local_season=True
    )


def create_local_season_year_panel_figures(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    dpi: int
) -> list[Path]:
    """Create year panels for complete hemisphere-aligned calendar years."""
    aligned_start_year = start_year + 1
    if aligned_start_year > end_year:
        raise ValueError(
            "At least two source years are required to create complete "
            "hemisphere-aligned year panels."
        )

    aligned = add_local_season_coordinates(panel)
    aligned[YEAR_COLUMN] = aligned["local_year"]
    aligned[MONTH_COLUMN] = aligned["local_month"]
    LOGGER.info(
        "[LOCAL SEASON] Year panels use aligned years %s-%s. "
        "The first source year is omitted to avoid partial aligned years.",
        aligned_start_year,
        end_year
    )
    return create_year_panel_figures(
        aligned,
        output_dir,
        aligned_start_year,
        end_year,
        dpi,
        local_season=True
    )

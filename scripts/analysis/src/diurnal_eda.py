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


def aggregate_sensor_city_months(
    panel: pd.DataFrame,
    sensor: str,
    start_year: int,
    end_year: int,
    aggregation: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate paired day/night SUHI values by city and calendar month."""
    if sensor not in SENSOR_COLUMNS:
        raise ValueError(f"Unsupported sensor: {sensor}")
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be 'mean' or 'median'.")

    sensor_columns = SENSOR_COLUMNS[sensor]
    required_columns = [
        *KEY_COLUMNS,
        sensor_columns["day"],
        sensor_columns["night"],
        sensor_columns["qc"]
    ]
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

    source_summary = (
        in_period
        .groupby(MONTH_COLUMN)
        .agg(
            total_city_year_months=(CITY_COLUMN, "size"),
            total_cities=(CITY_COLUMN, "nunique")
        )
        .reindex(range(1, 13), fill_value=0)
    )

    paired_pass = (
        in_period[sensor_columns["qc"]].fillna(False)
        & in_period[sensor_columns["day"]].notna()
        & in_period[sensor_columns["night"]].notna()
    )
    valid = in_period.loc[paired_pass].copy()
    valid_summary = (
        valid
        .groupby(MONTH_COLUMN)
        .agg(
            qc_pass_city_year_months=(CITY_COLUMN, "size"),
            qc_pass_cities=(CITY_COLUMN, "nunique")
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
    plotted_cities = (
        city_month
        .groupby(MONTH_COLUMN)[CITY_COLUMN]
        .nunique()
        .reindex(range(1, 13), fill_value=0)
    )
    summary["plotted_cities"] = plotted_cities.astype("int64")
    summary.index.name = MONTH_COLUMN
    return city_month, summary


def print_sensor_summary(
    sensor: str,
    summary: pd.DataFrame,
    start_year: int,
    end_year: int
) -> None:
    """Print QC coverage and plotted-city counts for each calendar month."""
    print(
        f"[EDA] {sensor.upper()} paired day/night QC, "
        f"{start_year}-{end_year}"
    )
    print(
        "month  passed_city_year_months/total  pass_rate  "
        "cities_with_pass/total  plotted_cities"
    )
    for month, row in summary.iterrows():
        print(
            f"{month:02d}     "
            f"{int(row['qc_pass_city_year_months']):,}/"
            f"{int(row['total_city_year_months']):,}  "
            f"{row['qc_pass_rate_pct']:.1f}%      "
            f"{int(row['qc_pass_cities']):,}/"
            f"{int(row['total_cities']):,}  "
            f"{int(row['plotted_cities']):,}"
        )


def create_diurnal_sensor_figures(
    city_months: dict[str, pd.DataFrame],
    output_dir: Path,
    start_year: int,
    end_year: int,
    aggregation: str,
    dpi: int
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

    lower = min(values)
    upper = max(values)
    padding = max((upper - lower) * 0.05, 0.25)
    lower -= padding
    upper += padding
    output_dir.mkdir(parents=True, exist_ok=True)

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
                axis.scatter(
                    month_frame["day_suhi"],
                    month_frame["night_suhi"],
                    s=18,
                    alpha=0.6,
                    color=SENSOR_COLORS[sensor],
                    edgecolors="none"
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
            axis.plot(
                [lower, upper],
                [lower, upper],
                linestyle="--",
                linewidth=0.9,
                color="0.35"
            )
            axis.set_title(MONTH_NAMES[month - 1])
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2, linewidth=0.5)

        figure.suptitle(
            f"{sensor.title()} daytime vs nighttime SUHI "
            f"({aggregation} by city-month, {start_year}-{end_year})",
            fontsize=15
        )
        figure.supxlabel("Daytime SUHI (°C)")
        figure.supylabel("Nighttime SUHI (°C)")
        figure.legend(
            handles=[
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
            ],
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            frameon=False
        )
        figure.tight_layout(rect=(0.04, 0.04, 0.96, 0.92))

        output_path = output_dir / (
            f"{sensor}_day_vs_night_suhi_{aggregation}_"
            f"{start_year}-{end_year}.png"
        )
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths

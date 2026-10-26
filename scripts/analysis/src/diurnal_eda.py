import logging
from math import isfinite

import pandas as pd


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
LATITUDE_GROUP_ORDER = (
    "Northern Hemisphere",
    "Southern Hemisphere",
    "Equatorial Belt"
)
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
    if "latitude_group" in panel.columns:
        required_columns.append("latitude_group")
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
    return in_period, in_period.loc[paired_pass].copy()


def _add_latitude_group_to_city_months(
    city_months: pd.DataFrame,
    valid: pd.DataFrame
) -> pd.DataFrame:
    if "latitude_group" not in valid.columns:
        return city_months
    city_groups = (
        valid.groupby(CITY_COLUMN)["latitude_group"]
        .nunique(dropna=False)
    )
    if city_groups.gt(1).any():
        raise ValueError("A city has inconsistent latitude group labels.")
    city_groups = (
        valid.groupby(CITY_COLUMN)["latitude_group"]
        .first()
        .rename("latitude_group")
        .reset_index()
    )
    return city_months.merge(
        city_groups,
        on=CITY_COLUMN,
        validate="many_to_one"
    )


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

    city_months = (
        valid
        .groupby([CITY_COLUMN, MONTH_COLUMN], sort=True)
        .agg(
            day_suhi=(sensor_columns["day"], aggregation),
            night_suhi=(sensor_columns["night"], aggregation),
            n_valid_years=(YEAR_COLUMN, "nunique")
        )
        .reset_index()
    )
    city_months = _add_latitude_group_to_city_months(city_months, valid)
    plotted_cities = (
        city_months
        .groupby(MONTH_COLUMN)[CITY_COLUMN]
        .nunique()
        .reindex(range(1, 13), fill_value=0)
    )
    summary["plotted_cities"] = plotted_cities.astype("int64")
    summary.index.name = MONTH_COLUMN
    return city_months, summary


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
    if "latitude_group" in valid.columns:
        plotted_rows["latitude_group"] = valid["latitude_group"]
    return plotted_rows, summary


def add_latitude_groups(
    panel: pd.DataFrame,
    equatorial_band_degrees: float
) -> pd.DataFrame:
    """Assign cities to northern, southern, or equatorial latitude groups."""
    if (
        isinstance(equatorial_band_degrees, bool)
        or not isinstance(equatorial_band_degrees, (int, float))
        or not isfinite(equatorial_band_degrees)
        or not 0 <= equatorial_band_degrees <= 90
    ):
        raise ValueError(
            "equatorial_band_degrees must be a finite number between 0 and 90."
        )
    if "centroid_lat" not in panel.columns:
        raise ValueError("Latitude grouping requires a centroid_lat column.")

    grouped = panel.copy()
    centroid_latitude = pd.to_numeric(
        grouped["centroid_lat"],
        errors="raise"
    )
    if centroid_latitude.isna().any() or not centroid_latitude.between(
        -90,
        90
    ).all():
        raise ValueError("centroid_lat values must be finite and in [-90, 90].")

    southern = centroid_latitude.lt(-equatorial_band_degrees)
    northern = centroid_latitude.gt(equatorial_band_degrees)
    grouped["latitude_group"] = "Equatorial Belt"
    grouped.loc[northern, "latitude_group"] = "Northern Hemisphere"
    grouped.loc[southern, "latitude_group"] = "Southern Hemisphere"
    return grouped


def add_local_season_coordinates(
    panel: pd.DataFrame,
    equatorial_band_degrees: float
) -> pd.DataFrame:
    """Shift southern observations and derive their aligned season year."""
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

    aligned = add_latitude_groups(panel, equatorial_band_degrees)
    southern = aligned["latitude_group"].eq("Southern Hemisphere")

    month_index = (
        aligned[MONTH_COLUMN] - 1
        + southern.astype("int64") * 6
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
    aggregation: str,
    equatorial_band_degrees: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate sensor data by latitude-aligned local month."""
    aligned = add_local_season_coordinates(
        panel,
        equatorial_band_degrees
    )
    source_period = aligned.loc[
        aligned[YEAR_COLUMN].between(start_year, end_year)
    ].copy()
    source_period[MONTH_COLUMN] = source_period["local_month"]
    return aggregate_sensor_city_months(
        source_period,
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
        "[EDA] %s paired day/night QC, %s-%s",
        sensor.upper(),
        start_year,
        end_year
    )
    LOGGER.info(
        "month  passed_city_year_months/total  pass_rate  "
        "cities_with_pass/total  plotted_cities"
    )
    for month, row in summary.iterrows():
        LOGGER.info(
            "%02d     %s/%s  %.1f%%      %s/%s  %s",
            month,
            f"{int(row['qc_pass_city_year_months']):,}",
            f"{int(row['total_city_year_months']):,}",
            row["qc_pass_rate_pct"],
            f"{int(row['qc_pass_cities']):,}",
            f"{int(row['total_cities']):,}",
            f"{int(row['plotted_cities']):,}"
        )


def log_year_month_summary(
    sensor: str,
    month: int,
    summary: pd.DataFrame,
    start_year: int,
    end_year: int,
    group_name: str | None = None
) -> None:
    """Log yearly QC counts for one sensor, month, and optional region."""
    label = f"{sensor.upper()} {MONTH_NAMES[month - 1]}"
    if group_name is not None:
        label = f"{label} {group_name}"
    LOGGER.info("[EDA YEAR] %s (%s-%s)", label, start_year, end_year)
    LOGGER.info("year  passed_city_months/total  pass_rate  cities_with_pass")
    for year in range(start_year, end_year + 1):
        row = summary.loc[(year, month)]
        LOGGER.info(
            "%s  %s/%s  %.1f%%  %s",
            year,
            f"{int(row['qc_pass_city_months']):,}",
            f"{int(row['total_city_months']):,}",
            row["qc_pass_rate_pct"],
            f"{int(row['qc_pass_cities']):,}"
        )

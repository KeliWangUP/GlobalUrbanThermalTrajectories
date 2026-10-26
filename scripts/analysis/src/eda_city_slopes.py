import numpy as np
import pandas as pd

from src.diurnal_eda import (
    CITY_COLUMN,
    KEY_COLUMNS,
    LATITUDE_GROUP_ORDER,
    MONTH_COLUMN,
    SENSOR_COLUMNS,
    YEAR_COLUMN,
    add_latitude_groups
)


def calculate_city_month_slopes(
    panel: pd.DataFrame,
    sensor: str,
    start_year: int,
    end_year: int,
    equatorial_band_degrees: float,
    minimum_years: int,
    climate_groups: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit paired-QC city/month D and N slopes, scaled to SUHI per decade."""
    if sensor not in SENSOR_COLUMNS:
        raise ValueError(f"Unsupported sensor: {sensor}")
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")
    if isinstance(minimum_years, bool) or not isinstance(minimum_years, int):
        raise ValueError("minimum_years must be an integer.")
    if minimum_years < 2:
        raise ValueError("minimum_years must be at least 2.")
    if minimum_years > end_year - start_year + 1:
        raise ValueError(
            "minimum_years cannot exceed the configured number of years."
        )
    if not climate_groups or len(climate_groups) != len(set(climate_groups)):
        raise ValueError("climate_groups must contain unique group names.")

    sensor_columns = SENSOR_COLUMNS[sensor]
    required_columns = [
        *KEY_COLUMNS,
        "centroid_lat",
        "koppen_geiger_macro_group",
        sensor_columns["day"],
        sensor_columns["night"],
        sensor_columns["qc"]
    ]
    missing = [column for column in required_columns if column not in panel]
    if missing:
        raise ValueError(f"Monthly SUHI panel is missing columns: {missing}")
    if panel.duplicated(KEY_COLUMNS).any():
        raise ValueError("Monthly SUHI panel contains duplicate city/month keys.")

    period = panel.loc[
        panel[YEAR_COLUMN].between(start_year, end_year),
        required_columns
    ].copy()
    if period.empty:
        raise ValueError(
            f"No monthly SUHI records found for {start_year}-{end_year}."
        )
    if not period[MONTH_COLUMN].between(1, 12).all():
        raise ValueError("Month values in the monthly SUHI panel must be 1..12.")
    if not pd.api.types.is_bool_dtype(period[sensor_columns["qc"]].dtype):
        raise TypeError(
            f"{sensor_columns['qc']} must contain boolean QC flags."
        )
    unexpected_groups = (
        set(period["koppen_geiger_macro_group"].dropna().unique())
        - set(climate_groups)
    )
    if unexpected_groups:
        raise ValueError(
            "Monthly panel has climate macro-groups missing from config: "
            f"{sorted(unexpected_groups)}"
        )

    period = add_latitude_groups(period, equatorial_band_degrees)
    city_group_counts = period.groupby(CITY_COLUMN)[
        "latitude_group"
    ].nunique(dropna=False)
    if city_group_counts.gt(1).any():
        raise ValueError("A city has inconsistent latitude group labels.")
    city_climate_counts = period.groupby(CITY_COLUMN)[
        "koppen_geiger_macro_group"
    ].nunique(dropna=False)
    if city_climate_counts.gt(1).any():
        raise ValueError("A city has inconsistent climate macro-group labels.")

    value_columns = [sensor_columns["day"], sensor_columns["night"]]
    values = period[value_columns].apply(pd.to_numeric, errors="raise")
    qc_pass = period[sensor_columns["qc"]].fillna(False)
    valid = period.loc[
        qc_pass & values.notna().all(axis=1)
    ].copy()
    valid[value_columns] = values.loc[valid.index].astype("float64")
    if not np.isfinite(valid[value_columns].to_numpy()).all():
        raise ValueError("QC-passing SUHI values must all be finite.")

    if valid.empty:
        raise ValueError(
            f"No QC-passing {sensor} day/night observations found for "
            f"{start_year}-{end_year}."
        )

    records = []
    group_columns = ["latitude_group", MONTH_COLUMN]
    for (latitude_group, month), month_rows in valid.groupby(
        group_columns,
        sort=True
    ):
        for city_id, city_rows in month_rows.groupby(CITY_COLUMN, sort=True):
            city_rows = city_rows.sort_values(YEAR_COLUMN, kind="stable")
            year_count = city_rows[YEAR_COLUMN].nunique()
            if year_count < minimum_years:
                continue
            years = city_rows[YEAR_COLUMN].to_numpy(dtype="float64")
            centered_years = years - years.mean()
            denominator = np.dot(centered_years, centered_years)
            if denominator <= 0:
                raise ValueError(
                    f"City {city_id} month {month} has no year variation."
                )
            day = city_rows[sensor_columns["day"]].to_numpy(dtype="float64")
            night = city_rows[sensor_columns["night"]].to_numpy(
                dtype="float64"
            )
            beta_day = np.dot(centered_years, day - day.mean()) / denominator
            beta_night = (
                np.dot(centered_years, night - night.mean()) / denominator
            )
            records.append({
                CITY_COLUMN: city_id,
                "latitude_group": latitude_group,
                "koppen_geiger_macro_group": (
                    city_rows["koppen_geiger_macro_group"].iloc[0]
                ),
                MONTH_COLUMN: month,
                "n_valid_years": year_count,
                "beta_day_per_decade": beta_day * 10,
                "beta_night_per_decade": beta_night * 10
            })

    slopes = pd.DataFrame.from_records(
        records,
        columns=[
            CITY_COLUMN,
            "latitude_group",
            "koppen_geiger_macro_group",
            MONTH_COLUMN,
            "n_valid_years",
            "beta_day_per_decade",
            "beta_night_per_decade"
        ]
    )
    if not slopes.empty:
        slopes = slopes.sort_values(
            ["latitude_group", MONTH_COLUMN, CITY_COLUMN],
            kind="stable"
        ).reset_index(drop=True)

    eligible_counts = (
        valid.groupby(["latitude_group", MONTH_COLUMN])[CITY_COLUMN]
        .nunique()
        .rename("n_cities_with_qc_data")
    )
    plotted_counts = (
        slopes.groupby(["latitude_group", MONTH_COLUMN])[CITY_COLUMN]
        .nunique()
        .rename("n_cities_plotted")
    )
    all_groups = pd.MultiIndex.from_product(
        [LATITUDE_GROUP_ORDER, range(1, 13)],
        names=["latitude_group", MONTH_COLUMN]
    )
    coverage = pd.concat(
        [eligible_counts, plotted_counts],
        axis=1
    ).reindex(all_groups, fill_value=0).reset_index()
    coverage[["n_cities_with_qc_data", "n_cities_plotted"]] = (
        coverage[["n_cities_with_qc_data", "n_cities_plotted"]]
        .fillna(0)
        .astype("int64")
    )
    coverage["n_cities_below_minimum"] = (
        coverage["n_cities_with_qc_data"] - coverage["n_cities_plotted"]
    )
    return slopes, coverage

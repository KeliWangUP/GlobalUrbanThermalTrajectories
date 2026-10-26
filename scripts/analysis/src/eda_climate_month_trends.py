import numpy as np
import pandas as pd

from src.diurnal_eda import (
    CITY_COLUMN,
    KEY_COLUMNS,
    MONTH_COLUMN,
    SENSOR_COLUMNS,
    YEAR_COLUMN,
    add_local_season_coordinates
)
from src.eda_trends import ASYMMETRY_COLUMNS


TREND_METRICS = ("day", "night", "asymmetry")


def _bootstrap_medians(
    values: np.ndarray,
    replicates: int,
    confidence_level: float,
    random_generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = len(values)
    bootstrap_values = np.empty((replicates, values.shape[1]))
    batch_size = 128
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = random_generator.integers(
            0,
            sample_count,
            size=(stop - start, sample_count)
        )
        bootstrap_values[start:stop] = np.median(values[indices], axis=1)

    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(
        bootstrap_values,
        [tail, 1 - tail],
        axis=0
    )
    return lower, upper


def _prepare_city_slopes(
    panel: pd.DataFrame,
    sensor: str,
    start_year: int,
    end_year: int,
    equatorial_band_degrees: float,
    minimum_years: int,
    climate_groups: tuple[str, ...]
) -> pd.DataFrame:
    if sensor not in SENSOR_COLUMNS:
        raise ValueError(f"Unsupported sensor: {sensor}")
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")
    if (
        isinstance(minimum_years, bool)
        or not isinstance(minimum_years, int)
        or minimum_years < 2
    ):
        raise ValueError("minimum_years must be an integer of at least 2.")
    if minimum_years > end_year - start_year + 1:
        raise ValueError(
            "minimum_years cannot exceed the configured number of years."
        )
    if not climate_groups or len(climate_groups) != len(set(climate_groups)):
        raise ValueError("climate_groups must contain unique group names.")

    sensor_columns = SENSOR_COLUMNS[sensor]
    asymmetry_column = ASYMMETRY_COLUMNS[sensor]
    climate_column = "koppen_geiger_macro_group"
    required_columns = [
        *KEY_COLUMNS,
        "centroid_lat",
        climate_column,
        sensor_columns["qc"],
        sensor_columns["day"],
        sensor_columns["night"],
        asymmetry_column
    ]
    missing = [column for column in required_columns if column not in panel]
    if missing:
        raise ValueError(f"Monthly SUHI panel is missing columns: {missing}")
    if panel.duplicated(KEY_COLUMNS).any():
        raise ValueError("Monthly SUHI panel contains duplicate city/month keys.")

    source = panel.loc[
        panel[YEAR_COLUMN].between(start_year, end_year),
        required_columns
    ].copy()
    if source.empty:
        raise ValueError(
            f"No monthly SUHI records found for {start_year}-{end_year}."
        )
    if not source[MONTH_COLUMN].between(1, 12).all():
        raise ValueError("Month values in the monthly SUHI panel must be 1..12.")
    if not pd.api.types.is_bool_dtype(source[sensor_columns["qc"]].dtype):
        raise TypeError(
            f"{sensor_columns['qc']} must contain boolean QC flags."
        )
    unexpected_groups = (
        set(source[climate_column].dropna().unique()) - set(climate_groups)
    )
    if unexpected_groups:
        raise ValueError(
            "Monthly panel has climate macro-groups missing from config: "
            f"{sorted(unexpected_groups)}"
        )
    city_climate_counts = source.groupby(CITY_COLUMN)[
        climate_column
    ].nunique(dropna=False)
    if city_climate_counts.gt(1).any():
        raise ValueError("A city has inconsistent climate macro-group labels.")

    aligned = add_local_season_coordinates(
        source,
        equatorial_band_degrees
    )
    aligned[YEAR_COLUMN] = aligned["local_year"]
    aligned[MONTH_COLUMN] = aligned["local_month"]
    aligned = aligned.loc[aligned[climate_column].notna()].copy()

    value_columns = [
        sensor_columns["day"],
        sensor_columns["night"],
        asymmetry_column
    ]
    numeric_values = aligned[value_columns].apply(
        pd.to_numeric,
        errors="raise"
    )
    qc_pass = aligned[sensor_columns["qc"]].fillna(False)
    valid = aligned.loc[
        qc_pass & numeric_values.notna().all(axis=1)
    ].copy()
    valid[value_columns] = numeric_values.loc[valid.index].astype("float64")
    if not np.isfinite(valid[value_columns].to_numpy()).all():
        raise ValueError(
            f"QC-passing {sensor} D/N/A values must all be finite."
        )
    if valid.empty:
        raise ValueError(
            f"No QC-passing {sensor} climate-group D/N/A values found."
        )

    records = []
    metric_columns = ("day", "night", "asymmetry")
    group_columns = [climate_column, MONTH_COLUMN]
    for (climate_group, month), month_rows in valid.groupby(
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
            record = {
                CITY_COLUMN: city_id,
                climate_column: climate_group,
                MONTH_COLUMN: month,
                "n_valid_years": year_count
            }
            for metric, column in zip(metric_columns, value_columns):
                observations = city_rows[column].to_numpy(dtype="float64")
                slope = np.dot(
                    centered_years,
                    observations - observations.mean()
                ) / denominator
                record[f"beta_{metric}_per_decade"] = slope * 10
            records.append(record)

    columns = [
        CITY_COLUMN,
        climate_column,
        MONTH_COLUMN,
        "n_valid_years",
        *(f"beta_{metric}_per_decade" for metric in metric_columns)
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def summarize_climate_month_slopes(
    panel: pd.DataFrame,
    sensor: str,
    start_year: int,
    end_year: int,
    equatorial_band_degrees: float,
    minimum_years: int,
    climate_groups: tuple[str, ...],
    bootstrap_replicates: int,
    confidence_level: float,
    random_seed: int
) -> pd.DataFrame:
    """Summarize city trend slopes by macroclimate group and local-season month."""
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 100
    ):
        raise ValueError("bootstrap_replicates must be an integer of at least 100.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or not 0 <= random_seed <= 2**32 - 1
    ):
        raise ValueError("random_seed must be an integer in [0, 2**32 - 1].")

    climate_column = "koppen_geiger_macro_group"
    slopes = _prepare_city_slopes(
        panel,
        sensor,
        start_year,
        end_year,
        equatorial_band_degrees,
        minimum_years,
        climate_groups
    )
    generator = np.random.default_rng(random_seed)
    index = pd.MultiIndex.from_product(
        [climate_groups, range(1, 13)],
        names=[climate_column, MONTH_COLUMN]
    )
    records = []
    beta_columns = [f"beta_{metric}_per_decade" for metric in TREND_METRICS]
    for (climate_group, month), city_rows in slopes.groupby(
        [climate_column, MONTH_COLUMN],
        sort=False
    ):
        values = city_rows[beta_columns].to_numpy(dtype="float64")
        medians = np.median(values, axis=0)
        if len(values) == 1:
            lower = medians
            upper = medians
        else:
            lower, upper = _bootstrap_medians(
                values,
                bootstrap_replicates,
                confidence_level,
                generator
            )
        record = {
            climate_column: climate_group,
            MONTH_COLUMN: month,
            "n_cities": city_rows[CITY_COLUMN].nunique()
        }
        for metric_index, metric in enumerate(TREND_METRICS):
            record[f"median_beta_{metric}"] = medians[metric_index]
            record[f"ci_lower_beta_{metric}"] = lower[metric_index]
            record[f"ci_upper_beta_{metric}"] = upper[metric_index]
        records.append(record)

    summary_columns = [
        climate_column,
        MONTH_COLUMN,
        "n_cities",
        *[
            column
            for metric in TREND_METRICS
            for column in (
                f"median_beta_{metric}",
                f"ci_lower_beta_{metric}",
                f"ci_upper_beta_{metric}"
            )
        ]
    ]
    if records:
        summary = (
            pd.DataFrame.from_records(records, columns=summary_columns)
            .set_index([climate_column, MONTH_COLUMN])
            .reindex(index)
            .reset_index()
        )
    else:
        summary = pd.DataFrame(index=index).reset_index()
        for column in summary_columns[2:]:
            summary[column] = np.nan
    summary["n_cities"] = summary["n_cities"].fillna(0).astype("int64")
    return summary

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


ASYMMETRY_COLUMNS = {"terra": "A_T", "aqua": "A_A"}
MEAN_COLUMNS = {"terra": "M_T", "aqua": "M_A"}
TREND_METRICS = ("day", "night", "asymmetry", "mean")
BOOTSTRAP_BATCH_SIZE = 128


def _bootstrap_median_intervals(
    values: np.ndarray,
    replicates: int,
    confidence_level: float,
    random_generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = values.shape[0]
    bootstrap_medians = np.empty((replicates, values.shape[1]))
    for start in range(0, replicates, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, replicates)
        sample_indices = random_generator.integers(
            0,
            sample_count,
            size=(stop - start, sample_count)
        )
        bootstrap_medians[start:stop] = np.median(
            values[sample_indices],
            axis=1
        )

    tail_probability = (1 - confidence_level) / 2
    lower, upper = np.quantile(
        bootstrap_medians,
        [tail_probability, 1 - tail_probability],
        axis=0
    )
    return lower, upper


def annual_sensor_trend_summary(
    panel: pd.DataFrame,
    sensor: str,
    start_year: int,
    end_year: int,
    equatorial_band_degrees: float,
    bootstrap_replicates: int,
    confidence_level: float,
    random_seed: int
) -> pd.DataFrame:
    """Summarize yearly city medians and bootstrap CIs by month and latitude."""
    if sensor not in SENSOR_COLUMNS:
        raise ValueError(f"Unsupported sensor: {sensor}")
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 100
    ):
        raise ValueError("bootstrap_replicates must be at least 100.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or not 0 <= random_seed <= 2**32 - 1
    ):
        raise ValueError("random_seed must be an integer in [0, 2**32 - 1].")

    sensor_columns = SENSOR_COLUMNS[sensor]
    metric_columns = [
        sensor_columns["day"],
        sensor_columns["night"],
        ASYMMETRY_COLUMNS[sensor],
        MEAN_COLUMNS[sensor]
    ]
    required_columns = [
        *KEY_COLUMNS,
        "centroid_lat",
        sensor_columns["qc"],
        *metric_columns
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

    period = add_latitude_groups(period, equatorial_band_degrees)
    passed = period[sensor_columns["qc"]].fillna(False)
    has_values = period[metric_columns].notna().all(axis=1)
    valid = period.loc[passed & has_values].copy()
    values = valid[metric_columns].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError(
            f"QC-passing {sensor} D/N/A/M values must all be finite."
        )
    valid.loc[:, metric_columns] = values
    valid = valid.sort_values(
        [YEAR_COLUMN, MONTH_COLUMN, "latitude_group", CITY_COLUMN],
        kind="stable"
    )

    random_generator = np.random.default_rng(random_seed)
    result_columns = [
        YEAR_COLUMN,
        MONTH_COLUMN,
        "latitude_group",
        "n_cities",
        *[
            column
            for metric in TREND_METRICS
            for column in (
                f"median_{metric}",
                f"ci_lower_{metric}",
                f"ci_upper_{metric}"
            )
        ]
    ]
    records = []
    group_columns = [YEAR_COLUMN, MONTH_COLUMN, "latitude_group"]
    for group_key, city_rows in valid.groupby(group_columns, sort=True):
        group_values = city_rows[metric_columns].to_numpy(dtype="float64")
        medians = np.median(group_values, axis=0)
        if len(city_rows) == 1:
            lower = medians
            upper = medians
        else:
            lower, upper = _bootstrap_median_intervals(
                group_values,
                bootstrap_replicates,
                confidence_level,
                random_generator
            )

        record = dict(zip(group_columns, group_key))
        record["n_cities"] = city_rows[CITY_COLUMN].nunique()
        for index, metric in enumerate(TREND_METRICS):
            record[f"median_{metric}"] = medians[index]
            record[f"ci_lower_{metric}"] = lower[index]
            record[f"ci_upper_{metric}"] = upper[index]
        records.append(record)

    complete_index = pd.MultiIndex.from_product(
        [
            range(start_year, end_year + 1),
            range(1, 13),
            LATITUDE_GROUP_ORDER
        ],
        names=group_columns
    )
    summary = (
        pd.DataFrame.from_records(records, columns=result_columns)
        .set_index(group_columns)
        .reindex(complete_index)
        .reset_index()
    )
    summary["n_cities"] = summary["n_cities"].fillna(0).astype("int64")
    return summary

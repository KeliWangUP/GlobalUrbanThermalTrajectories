from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.diurnal_eda import CITY_COLUMN, KEY_COLUMNS, YEAR_COLUMN


DATASETS = ("urban", "rural", "suhi")
METRICS = ("D", "N", "A")
SENSORS = ("terra", "aqua")
CLIMATE_COLUMN = "koppen_geiger_macro_group"
LATITUDE_GROUP_COLUMN = "latitude_group"
SEASON_KEYS = (
    CITY_COLUMN,
    "season_year",
    "season",
    "centroid_lat",
    LATITUDE_GROUP_COLUMN,
    CLIMATE_COLUMN
)


@dataclass(frozen=True)
class SeasonalContrastResults:
    city_month_labels: pd.DataFrame
    city_year_seasons: pd.DataFrame
    city_year_contrasts: pd.DataFrame
    city_trends: pd.DataFrame
    climate_summary: pd.DataFrame
    climate_season_summary: pd.DataFrame


def _validate_months(
    summer_months: tuple[int, ...],
    winter_months: tuple[int, ...]
) -> None:
    if (
        len(summer_months) != 3
        or len(winter_months) != 3
        or len(set(summer_months)) != 3
        or len(set(winter_months)) != 3
        or set(summer_months) & set(winter_months)
        or any(month not in range(1, 13) for month in summer_months)
        or any(month not in range(1, 13) for month in winter_months)
    ):
        raise ValueError(
            "Summer and winter must each contain three unique, non-overlapping "
            "calendar months in 1..12."
        )


def _build_city_month_labels(
    panel: pd.DataFrame,
    start_year: int,
    end_year: int,
    equatorial_band_degrees: float,
    northern_summer_months: tuple[int, ...],
    northern_winter_months: tuple[int, ...],
    climate_groups: tuple[str, ...]
) -> pd.DataFrame:
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")
    if not 0 <= equatorial_band_degrees <= 90:
        raise ValueError("equatorial_band_degrees must be between 0 and 90.")
    _validate_months(northern_summer_months, northern_winter_months)

    required = [*KEY_COLUMNS, "centroid_lat", CLIMATE_COLUMN]
    missing = [column for column in required if column not in panel]
    if missing:
        raise ValueError(f"Monthly SUHI panel is missing columns: {missing}")
    if panel.duplicated(KEY_COLUMNS).any():
        raise ValueError("Monthly SUHI panel contains duplicate city/month keys.")

    period = panel.loc[
        panel[YEAR_COLUMN].between(start_year - 1, end_year),
        required
    ].copy()
    if period.empty:
        raise ValueError(
            f"No monthly SUHI records found for {start_year}-{end_year}."
        )
    if not period["month"].between(1, 12).all():
        raise ValueError("Month values in the monthly SUHI panel must be 1..12.")

    latitude = pd.to_numeric(period["centroid_lat"], errors="raise")
    if (
        latitude.isna().any()
        or not np.isfinite(latitude.to_numpy(dtype="float64")).all()
        or not latitude.between(-90, 90).all()
    ):
        raise ValueError("City centroid latitudes must be finite and in [-90, 90].")
    period["centroid_lat"] = latitude

    unexpected_groups = (
        set(period[CLIMATE_COLUMN].dropna().unique()) - set(climate_groups)
    )
    if unexpected_groups:
        raise ValueError(
            "Monthly panel has climate macro-groups missing from config: "
            f"{sorted(unexpected_groups)}"
        )
    for column in ("centroid_lat", CLIMATE_COLUMN):
        city_counts = period.groupby(CITY_COLUMN)[column].nunique(
            dropna=False
        )
        if city_counts.gt(1).any():
            raise ValueError(f"A city has inconsistent {column} metadata.")

    north = period["centroid_lat"] > equatorial_band_degrees
    south = period["centroid_lat"] < -equatorial_band_degrees
    equatorial = ~(north | south)
    month = period["month"]
    northern_summer = month.isin(northern_summer_months)
    northern_winter = month.isin(northern_winter_months)

    is_summer = (
        (north & northern_summer) | (south & northern_winter)
    )
    is_winter = (
        (north & northern_winter) | (south & northern_summer)
    )
    period["latitude_group"] = np.select(
        [north, south],
        ["Northern Hemisphere", "Southern Hemisphere"],
        default="Equatorial Belt"
    )
    period["is_summer"] = is_summer.astype("int8")
    period["is_winter"] = is_winter.astype("int8")
    period["season"] = np.select(
        [equatorial, is_summer, is_winter],
        ["equatorial", "summer", "winter"],
        default="shoulder"
    )
    period["season_year"] = period[YEAR_COLUMN].astype("int64")
    december_season = (
        month.eq(12) & (period["is_summer"].eq(1) | period["is_winter"].eq(1))
    )
    period.loc[december_season, "season_year"] += 1
    return period.reset_index(drop=True)


def _sensor_season_values(
    labeled_panel: pd.DataFrame,
    sensor: str,
    minimum_valid_months: int
) -> pd.DataFrame:
    if sensor not in SENSORS:
        raise ValueError(f"Unsupported sensor: {sensor}")
    if (
        isinstance(minimum_valid_months, bool)
        or not isinstance(minimum_valid_months, int)
        or not 1 <= minimum_valid_months <= 3
    ):
        raise ValueError("minimum_valid_months must be an integer in 1..3.")

    suffix = "T" if sensor == "terra" else "A"
    qc_column = f"qc_pass_A_{suffix}"
    value_columns = [
        f"{side}_{sensor}_{slot}_mean"
        for side in ("urban", "rural", "suhi")
        for slot in ("day", "night")
    ]
    required = [*SEASON_KEYS, YEAR_COLUMN, "month", qc_column, *value_columns]
    missing = [column for column in required if column not in labeled_panel]
    if missing:
        raise ValueError(f"Monthly SUHI panel is missing columns: {missing}")
    if not pd.api.types.is_bool_dtype(labeled_panel[qc_column].dtype):
        raise TypeError(f"{qc_column} must contain boolean QC flags.")

    selected = labeled_panel.loc[
        labeled_panel["season"].isin(("summer", "winter")),
        required
    ].copy()
    numeric = selected[value_columns].apply(pd.to_numeric, errors="raise")
    qc_pass = selected[qc_column].fillna(False)
    finite = np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1)
    valid = qc_pass & numeric.notna().all(axis=1) & finite

    selected["_valid_month"] = valid.astype("int8")
    for column in value_columns:
        selected[column] = numeric[column].where(valid)

    output_columns = []
    for dataset in DATASETS:
        day_column = f"{dataset}_{sensor}_day_mean"
        night_column = f"{dataset}_{sensor}_night_mean"
        for metric, source_column in (
            ("D", day_column),
            ("N", night_column)
        ):
            output_column = f"{sensor}_{dataset}_{metric}"
            selected[output_column] = selected[source_column]
            output_columns.append(output_column)
        asymmetry_column = f"{sensor}_{dataset}_A"
        selected[asymmetry_column] = (
            selected[day_column] - selected[night_column]
        )
        output_columns.append(asymmetry_column)

    group_columns = [
        CITY_COLUMN,
        "season_year",
        "season",
        "centroid_lat",
        LATITUDE_GROUP_COLUMN,
        CLIMATE_COLUMN
    ]
    grouped = selected.groupby(
        group_columns,
        sort=True,
        dropna=False
    )
    means = grouped[output_columns].mean()
    counts = grouped["_valid_month"].sum().rename(
        f"n_valid_months_{sensor}"
    )
    result = means.join(counts).reset_index()
    incomplete = result[f"n_valid_months_{sensor}"] < minimum_valid_months
    result.loc[incomplete, output_columns] = np.nan
    return result


def _build_city_year_contrasts(
    city_year_seasons: pd.DataFrame,
    value_columns: list[str]
) -> pd.DataFrame:
    index_columns = [
        CITY_COLUMN,
        "season_year",
        "centroid_lat",
        LATITUDE_GROUP_COLUMN,
        CLIMATE_COLUMN
    ]
    index = pd.MultiIndex.from_frame(
        city_year_seasons[index_columns].drop_duplicates()
    )
    warm = (
        city_year_seasons.loc[city_year_seasons["season"].eq("summer")]
        .set_index(index_columns)[value_columns]
        .reindex(index)
    )
    cold = (
        city_year_seasons.loc[city_year_seasons["season"].eq("winter")]
        .set_index(index_columns)[value_columns]
        .reindex(index)
    )
    metadata = pd.DataFrame(index=index).reset_index()
    contrast = warm - cold
    contrast.columns = [f"contrast_{column}" for column in value_columns]
    return metadata.join(
        contrast.reset_index(drop=True)
    )


def _fit_city_contrast_trends(
    city_year_contrasts: pd.DataFrame,
    minimum_years: int
) -> pd.DataFrame:
    if (
        isinstance(minimum_years, bool)
        or not isinstance(minimum_years, int)
        or minimum_years < 2
    ):
        raise ValueError("minimum_years must be an integer of at least 2.")

    identity_columns = [
        CITY_COLUMN,
        "centroid_lat",
        LATITUDE_GROUP_COLUMN,
        CLIMATE_COLUMN
    ]
    records = []
    for sensor in SENSORS:
        contrast_columns = [
            f"contrast_{sensor}_{dataset}_{metric}"
            for dataset in DATASETS
            for metric in METRICS
        ]
        for identity, city_rows in city_year_contrasts.groupby(
            identity_columns,
            sort=True,
            dropna=False
        ):
            complete = city_rows.dropna(subset=contrast_columns)
            n_years = complete["season_year"].nunique()
            if n_years < minimum_years:
                continue
            years = complete["season_year"].to_numpy(dtype="float64")
            centered_years = years - years.mean()
            denominator = np.dot(centered_years, centered_years)
            if denominator <= 0:
                continue
            record = dict(zip(identity_columns, identity))
            record["sensor"] = sensor
            record["n_valid_contrast_years"] = n_years
            for dataset in DATASETS:
                for metric in METRICS:
                    column = f"contrast_{sensor}_{dataset}_{metric}"
                    observations = complete[column].to_numpy(dtype="float64")
                    slope = np.dot(
                        centered_years,
                        observations - observations.mean()
                    ) / denominator
                    record[f"gamma_{dataset}_{metric}"] = slope * 10
            records.append(record)

    columns = [
        *identity_columns,
        "sensor",
        "n_valid_contrast_years",
        *[
            f"gamma_{dataset}_{metric}"
            for dataset in DATASETS
            for metric in METRICS
        ]
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _bootstrap_median_interval(
    values: np.ndarray,
    replicates: int,
    confidence_level: float,
    generator: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = np.empty(replicates, dtype="float64")
    batch_size = 128
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indexes = generator.integers(
            0,
            len(values),
            size=(stop - start, len(values))
        )
        samples[start:stop] = np.median(values[indexes], axis=1)
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail])
    return float(lower), float(upper)


def _summarize_climate_trends(
    city_trends: pd.DataFrame,
    climate_groups: tuple[str, ...],
    bootstrap_replicates: int,
    confidence_level: float,
    random_seed: int
) -> pd.DataFrame:
    trend_rows = city_trends.copy()
    trend_rows["climate_group"] = (
        trend_rows[CLIMATE_COLUMN].fillna("Unclassified")
    )
    group_names = ["All cities", *climate_groups, "Unclassified"]
    group_names = list(dict.fromkeys(group_names))
    generator = np.random.default_rng(random_seed)
    records = []

    for sensor in SENSORS:
        sensor_rows = trend_rows.loc[trend_rows["sensor"].eq(sensor)]
        for group_name in group_names:
            if group_name == "All cities":
                group_rows = sensor_rows
            else:
                group_rows = sensor_rows.loc[
                    sensor_rows["climate_group"].eq(group_name)
                ]
            for dataset in DATASETS:
                for metric in METRICS:
                    values = group_rows[
                        f"gamma_{dataset}_{metric}"
                    ].dropna().to_numpy(dtype="float64")
                    if len(values):
                        median = float(np.median(values))
                        lower, upper = _bootstrap_median_interval(
                            values,
                            bootstrap_replicates,
                            confidence_level,
                            generator
                        )
                        positive_pct = float((values > 0).mean() * 100)
                        negative_pct = float((values < 0).mean() * 100)
                        zero_pct = float((values == 0).mean() * 100)
                        if len(values) < 2:
                            direction = "insufficient_cities"
                        elif lower > 0:
                            direction = "increasing"
                        elif upper < 0:
                            direction = "decreasing"
                        else:
                            direction = "no_clear_median_trend"
                    else:
                        median = lower = upper = np.nan
                        positive_pct = negative_pct = zero_pct = np.nan
                        direction = "no_eligible_cities"
                    records.append({
                        "climate_group": group_name,
                        "sensor": sensor,
                        "dataset": dataset,
                        "metric": metric,
                        "n_cities": len(values),
                        "median_gamma_per_decade": median,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "positive_gamma_pct": positive_pct,
                        "negative_gamma_pct": negative_pct,
                        "zero_gamma_pct": zero_pct,
                        "median_trend_direction": direction,
                        "confidence_level": confidence_level,
                        "bootstrap_replicates": bootstrap_replicates
                    })
    return pd.DataFrame.from_records(records)


def _fit_city_season_trends(
    city_year_seasons: pd.DataFrame,
    city_year_contrasts: pd.DataFrame,
    minimum_years: int
) -> pd.DataFrame:
    identity_columns = [
        CITY_COLUMN,
        "centroid_lat",
        LATITUDE_GROUP_COLUMN,
        CLIMATE_COLUMN
    ]
    records = []
    for sensor in SENSORS:
        contrast_columns = [
            f"contrast_{sensor}_{dataset}_{metric}"
            for dataset in DATASETS
            for metric in METRICS
        ]
        value_columns = [
            f"{sensor}_{dataset}_{metric}"
            for dataset in DATASETS
            for metric in METRICS
        ]
        for identity, city_rows in city_year_contrasts.groupby(
            identity_columns,
            sort=True,
            dropna=False
        ):
            complete = city_rows.dropna(subset=contrast_columns)
            year_values = np.sort(complete["season_year"].unique())
            if len(year_values) < minimum_years:
                continue
            years = year_values.astype("float64")
            centered_years = years - years.mean()
            denominator = np.dot(centered_years, centered_years)
            if denominator <= 0:
                continue

            city_seasons = city_year_seasons.loc[
                city_year_seasons[CITY_COLUMN].eq(identity[0])
                & city_year_seasons["season_year"].isin(year_values)
            ].set_index(["season_year", "season"])
            paired_index = pd.MultiIndex.from_product(
                [year_values, ("summer", "winter")],
                names=("season_year", "season")
            )
            city_seasons = city_seasons.reindex(paired_index)
            for season in ("summer", "winter"):
                seasonal_rows = city_seasons.xs(season, level="season")
                record = dict(zip(identity_columns, identity))
                record["sensor"] = sensor
                record["season"] = season
                record["n_valid_paired_years"] = len(years)
                for column in value_columns:
                    observations = seasonal_rows[column].to_numpy(
                        dtype="float64"
                    )
                    if not np.isfinite(observations).all():
                        raise ValueError(
                            "Seasonal LST trends require finite values for "
                            "every paired contrast year."
                        )
                    slope = np.dot(
                        centered_years,
                        observations - observations.mean()
                    ) / denominator
                    dataset_metric = column.removeprefix(f"{sensor}_")
                    record[f"gamma_{dataset_metric}"] = slope * 10
                records.append(record)

    columns = [
        *identity_columns,
        "sensor",
        "season",
        "n_valid_paired_years",
        *[
            f"gamma_{dataset}_{metric}"
            for dataset in DATASETS
            for metric in METRICS
        ]
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _summarize_climate_season_trends(
    city_season_trends: pd.DataFrame,
    climate_groups: tuple[str, ...],
    bootstrap_replicates: int,
    confidence_level: float,
    random_seed: int
) -> pd.DataFrame:
    trend_rows = city_season_trends.copy()
    trend_rows["climate_group"] = (
        trend_rows[CLIMATE_COLUMN].fillna("Unclassified")
    )
    group_names = list(dict.fromkeys([
        "All cities",
        *climate_groups,
        "Unclassified"
    ]))
    generator = np.random.default_rng(random_seed)
    records = []

    for sensor in SENSORS:
        sensor_rows = trend_rows.loc[trend_rows["sensor"].eq(sensor)]
        for group_name in group_names:
            if group_name == "All cities":
                group_rows = sensor_rows
            else:
                group_rows = sensor_rows.loc[
                    sensor_rows["climate_group"].eq(group_name)
                ]
            for dataset in DATASETS:
                for metric in METRICS:
                    for season in ("summer", "winter"):
                        values = group_rows.loc[
                            group_rows["season"].eq(season),
                            f"gamma_{dataset}_{metric}"
                        ].dropna().to_numpy(dtype="float64")
                        if len(values):
                            median = float(np.median(values))
                            lower, upper = _bootstrap_median_interval(
                                values,
                                bootstrap_replicates,
                                confidence_level,
                                generator
                            )
                        else:
                            median = lower = upper = np.nan
                        records.append({
                            "climate_group": group_name,
                            "sensor": sensor,
                            "dataset": dataset,
                            "metric": metric,
                            "season": season,
                            "n_cities": len(values),
                            "median_gamma_per_decade": median,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "confidence_level": confidence_level,
                            "bootstrap_replicates": bootstrap_replicates
                        })
    return pd.DataFrame.from_records(records)


def analyze_seasonal_contrasts(
    panel: pd.DataFrame,
    start_year: int,
    end_year: int,
    equatorial_band_degrees: float,
    northern_summer_months: tuple[int, ...],
    northern_winter_months: tuple[int, ...],
    minimum_valid_months_per_season: int,
    minimum_trend_years: int,
    climate_groups: tuple[str, ...],
    bootstrap_replicates: int,
    confidence_level: float,
    random_seed: int
) -> SeasonalContrastResults:
    """Analyze warm-minus-cold seasonal contrasts and their city-level trends."""
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 100
    ):
        raise ValueError("bootstrap_replicates must be an integer of at least 100.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if not climate_groups or len(climate_groups) != len(set(climate_groups)):
        raise ValueError("climate_groups must contain unique group names.")

    labels = _build_city_month_labels(
        panel=panel,
        start_year=start_year,
        end_year=end_year,
        equatorial_band_degrees=equatorial_band_degrees,
        northern_summer_months=northern_summer_months,
        northern_winter_months=northern_winter_months,
        climate_groups=climate_groups
    )
    source_columns = [
        *KEY_COLUMNS,
        *[
            f"qc_pass_A_{'T' if sensor == 'terra' else 'A'}"
            for sensor in SENSORS
        ],
        *[
            f"{side}_{sensor}_{slot}_mean"
            for side in DATASETS
            for sensor in SENSORS
            for slot in ("day", "night")
        ]
    ]
    missing = [column for column in source_columns if column not in panel]
    if missing:
        raise ValueError(f"Monthly SUHI panel is missing columns: {missing}")
    source = panel.loc[
        panel[YEAR_COLUMN].between(start_year - 1, end_year),
        source_columns
    ].merge(
        labels,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one"
    )
    source = source.loc[
        source["season_year"].between(start_year, end_year)
        & source["season"].isin(("summer", "winter"))
    ].copy()

    sensor_seasons = [
        _sensor_season_values(
            source,
            sensor,
            minimum_valid_months_per_season
        )
        for sensor in SENSORS
    ]
    merge_keys = list(SEASON_KEYS)
    city_year_seasons = sensor_seasons[0].merge(
        sensor_seasons[1],
        on=merge_keys,
        how="outer",
        validate="one_to_one"
    )
    city_year_seasons["is_summer"] = (
        city_year_seasons["season"].eq("summer").astype("int8")
    )
    city_year_seasons["is_winter"] = (
        city_year_seasons["season"].eq("winter").astype("int8")
    )
    value_columns = [
        f"{sensor}_{dataset}_{metric}"
        for sensor in SENSORS
        for dataset in DATASETS
        for metric in METRICS
    ]
    city_year_contrasts = _build_city_year_contrasts(
        city_year_seasons,
        value_columns
    )
    city_trends = _fit_city_contrast_trends(
        city_year_contrasts,
        minimum_trend_years
    )
    if city_trends.empty:
        raise ValueError(
            "No cities have enough complete warm/cold contrast years for "
            f"trend estimation (minimum={minimum_trend_years})."
        )
    city_season_trends = _fit_city_season_trends(
        city_year_seasons=city_year_seasons,
        city_year_contrasts=city_year_contrasts,
        minimum_years=minimum_trend_years
    )
    climate_season_summary = _summarize_climate_season_trends(
        city_season_trends=city_season_trends,
        climate_groups=climate_groups,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        random_seed=random_seed
    )
    climate_summary = _summarize_climate_trends(
        city_trends=city_trends,
        climate_groups=climate_groups,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        random_seed=random_seed
    )
    return SeasonalContrastResults(
        city_month_labels=labels,
        city_year_seasons=city_year_seasons,
        city_year_contrasts=city_year_contrasts,
        city_trends=city_trends,
        climate_summary=climate_summary,
        climate_season_summary=climate_season_summary
    )

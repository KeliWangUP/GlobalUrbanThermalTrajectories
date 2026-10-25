from pathlib import Path
import tempfile

import pandas as pd

from src.config import QualityControlConfig


KEY_COLUMNS = ["ID_UC_G0", "year", "month"]
LST_SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")
LST_MEAN_COLUMNS = [f"{slot}_mean" for slot in LST_SLOTS]
SUHI_COLUMNS = [f"suhi_{slot}_mean" for slot in LST_SLOTS]
DERIVED_SUHI_COLUMNS = ["A_T", "A_A", "M_T", "M_A"]
PANEL_VALUE_COLUMNS = [
    f"{side}_{slot}_mean"
    for slot in LST_SLOTS
    for side in ("urban", "rural")
] + SUHI_COLUMNS + DERIVED_SUHI_COLUMNS
VALID_MONTH_COUNT_COLUMNS = {
    **{
        f"suhi_{slot}_mean": f"n_valid_months_{slot}"
        for slot in LST_SLOTS
    },
    **{
        metric: f"n_valid_months_{metric}"
        for metric in DERIVED_SUHI_COLUMNS
    }
}
SEASON_BY_MONTH = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON"
}
SEASON_ORDER = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


def build_monthly_suhi(
    urban: pd.DataFrame,
    rural: pd.DataFrame,
    expected_year: int,
    quality_control: QualityControlConfig
) -> pd.DataFrame:
    """Join rural and urban data, apply slot-level QC, and calculate SUHI."""
    for name, frame in (("urban", urban), ("rural", rural)):
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(
                f"{name} data contain duplicate city/year/month keys "
                f"for {expected_year}."
            )
        years = frame["year"].unique()
        if len(years) != 1 or int(years[0]) != expected_year:
            raise ValueError(
                f"{name} parquet does not contain only year {expected_year}."
            )

    urban_metrics = [
        column for column in urban.columns if column not in KEY_COLUMNS
    ]
    rural_metrics = [
        column for column in rural.columns if column not in KEY_COLUMNS
    ]
    if urban_metrics != rural_metrics:
        raise ValueError(
            f"Urban/rural metric schemas differ for {expected_year}."
        )
    missing_means = [
        column for column in LST_MEAN_COLUMNS
        if column not in urban_metrics
    ]
    if missing_means:
        raise ValueError(
            f"Required LST mean columns are missing: {missing_means}"
        )
    required_qc_columns = [
        f"{slot}_{suffix}"
        for slot in LST_SLOTS
        for suffix in (
            "count",
            "clear_days_mean",
            "n8day_mean"
        )
    ]
    missing_qc_columns = [
        column for column in required_qc_columns
        if column not in urban_metrics
    ]
    if missing_qc_columns:
        raise ValueError(
            f"Required QC columns are missing: {missing_qc_columns}"
        )

    urban_for_join = urban.rename(
        columns={column: f"urban_{column}" for column in urban_metrics}
    )
    rural_for_join = rural.rename(
        columns={column: f"rural_{column}" for column in rural_metrics}
    )
    joined = rural_for_join.merge(
        urban_for_join,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True
    )
    unmatched = joined["_merge"] != "both"
    if unmatched.any():
        examples = joined.loc[unmatched, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(
            f"Rural records lack matching urban rows for {expected_year}: "
            f"{examples}"
        )
    joined = joined.drop(columns="_merge")

    for slot in LST_SLOTS:
        urban_mean_column = f"urban_{slot}_mean"
        rural_mean_column = f"rural_{slot}_mean"
        qc_pass = pd.Series(True, index=joined.index, dtype=bool)
        for side in ("urban", "rural"):
            qc_metrics = (
                (
                    f"{side}_{slot}_count",
                    quality_control.min_valid_pixels_per_side_slot
                ),
                (
                    f"{side}_{slot}_clear_days_mean",
                    quality_control.min_clear_days_mean_per_side_slot
                ),
                (
                    f"{side}_{slot}_n8day_mean",
                    quality_control.min_n8day_mean_per_side_slot
                )
            )
            for column, minimum in qc_metrics:
                qc_pass &= joined[column].ge(minimum).fillna(False)
        qc_pass &= joined[urban_mean_column].notna()
        qc_pass &= joined[rural_mean_column].notna()
        joined[f"qc_pass_{slot}"] = qc_pass

        joined.loc[
            ~qc_pass,
            [urban_mean_column, rural_mean_column]
        ] = float("nan")
        joined[f"suhi_{slot}_mean"] = (
            joined[urban_mean_column] - joined[rural_mean_column]
        )
    add_derived_suhi_metrics(joined, "qc_pass_")
    return joined.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def add_derived_suhi_metrics(
    panel: pd.DataFrame,
    qc_flag_prefix: str
) -> None:
    """Add day/night contrasts and means using QC-paired monthly SUHI values."""
    panel["A_T"] = (
        panel["suhi_terra_day_mean"] - panel["suhi_terra_night_mean"]
    )
    panel["A_A"] = (
        panel["suhi_aqua_day_mean"] - panel["suhi_aqua_night_mean"]
    )
    panel["M_T"] = (
        panel["suhi_terra_day_mean"] + panel["suhi_terra_night_mean"]
    ) / 2
    panel["M_A"] = (
        panel["suhi_aqua_day_mean"] + panel["suhi_aqua_night_mean"]
    ) / 2
    panel["qc_pass_A_T"] = (
        panel[f"{qc_flag_prefix}terra_day"]
        & panel[f"{qc_flag_prefix}terra_night"]
    )
    panel["qc_pass_A_A"] = (
        panel[f"{qc_flag_prefix}aqua_day"]
        & panel[f"{qc_flag_prefix}aqua_night"]
    )
    panel["qc_pass_M_T"] = panel["qc_pass_A_T"]
    panel["qc_pass_M_A"] = panel["qc_pass_A_A"]


def _aggregate_monthly_values(
    panel: pd.DataFrame,
    group_columns: list[str],
    expected_months: int,
    completeness_column: str,
    minimum_valid_months: int,
    quality_period: str
) -> pd.DataFrame:
    grouped = panel.groupby(group_columns, sort=True, observed=True)
    means = grouped[PANEL_VALUE_COLUMNS].mean().reset_index()
    valid_counts = (
        grouped[list(VALID_MONTH_COUNT_COLUMNS)]
        .count()
        .rename(columns=VALID_MONTH_COUNT_COLUMNS)
        .reset_index()
    )
    calendar_counts = (
        grouped["month"]
        .nunique()
        .rename("n_calendar_months")
        .reset_index()
    )
    result = means.merge(
        valid_counts,
        on=group_columns,
        validate="one_to_one"
    ).merge(
        calendar_counts,
        on=group_columns,
        validate="one_to_one"
    )
    result[completeness_column] = (
        result["n_calendar_months"] == expected_months
    )
    for slot in LST_SLOTS:
        qc_pass = (
            result[f"n_valid_months_{slot}"] >= minimum_valid_months
        )
        result[f"qc_pass_{quality_period}_{slot}"] = qc_pass
        for side in ("urban", "rural"):
            result.loc[~qc_pass, f"{side}_{slot}_mean"] = float("nan")
        result.loc[~qc_pass, f"suhi_{slot}_mean"] = float("nan")
    for metric in DERIVED_SUHI_COLUMNS:
        qc_pass = (
            result[f"n_valid_months_{metric}"] >= minimum_valid_months
        )
        result[f"qc_pass_{quality_period}_{metric}"] = qc_pass
        result.loc[~qc_pass, metric] = float("nan")
    return result


def aggregate_seasonal_suhi(
    monthly: pd.DataFrame,
    quality_control: QualityControlConfig
) -> pd.DataFrame:
    """Aggregate monthly values into fixed DJF/MAM/JJA/SON seasons."""
    seasonal = monthly.copy()
    seasonal["season"] = seasonal["month"].map(SEASON_BY_MONTH)
    seasonal["season_year"] = seasonal["year"]
    seasonal.loc[seasonal["month"] == 12, "season_year"] += 1

    result = _aggregate_monthly_values(
        panel=seasonal,
        group_columns=["ID_UC_G0", "season_year", "season"],
        expected_months=3,
        completeness_column="is_complete_season",
        minimum_valid_months=quality_control.min_valid_months_per_season,
        quality_period="season"
    )
    result["_season_order"] = result["season"].map(SEASON_ORDER)
    return (
        result
        .sort_values(
            ["ID_UC_G0", "season_year", "_season_order"],
            kind="stable"
        )
        .drop(columns="_season_order")
        .reset_index(drop=True)
    )


def aggregate_annual_suhi(
    monthly: pd.DataFrame,
    quality_control: QualityControlConfig
) -> pd.DataFrame:
    """Aggregate monthly values into calendar-year means."""
    return (
        _aggregate_monthly_values(
            panel=monthly,
            group_columns=["ID_UC_G0", "year"],
            expected_months=12,
            completeness_column="is_complete_year",
            minimum_valid_months=quality_control.min_valid_months_per_year,
            quality_period="year"
        )
        .sort_values(["ID_UC_G0", "year"], kind="stable")
        .reset_index(drop=True)
    )


def write_parquet_atomically(
    frame: pd.DataFrame,
    path: Path,
    compression: str
) -> None:
    """Write a Parquet file then replace the destination after success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_",
        suffix=".parquet",
        dir=path.parent,
        delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        frame.to_parquet(
            temporary_path,
            engine="pyarrow",
            compression=compression,
            index=False
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
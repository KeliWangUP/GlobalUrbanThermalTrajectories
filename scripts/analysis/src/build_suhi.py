from pathlib import Path
import tempfile

import pandas as pd


KEY_COLUMNS = ["ID_UC_G0", "year", "month"]
LST_SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")
LST_MEAN_COLUMNS = [f"{slot}_mean" for slot in LST_SLOTS]
PANEL_VALUE_COLUMNS = [
    f"{side}_{slot}_mean"
    for slot in LST_SLOTS
    for side in ("urban", "rural")
] + [f"suhi_{slot}_mean" for slot in LST_SLOTS]
SUHI_COLUMNS = [f"suhi_{slot}_mean" for slot in LST_SLOTS]
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
    expected_year: int
) -> pd.DataFrame:
    """Join the rural city subset to urban statistics and calculate monthly SUHI."""
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
        joined[f"suhi_{slot}_mean"] = (
            joined[f"urban_{slot}_mean"] - joined[f"rural_{slot}_mean"]
        )
    return joined.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def _aggregate_monthly_values(
    panel: pd.DataFrame,
    group_columns: list[str],
    expected_months: int,
    completeness_column: str
) -> pd.DataFrame:
    grouped = panel.groupby(group_columns, sort=True, observed=True)
    means = grouped[PANEL_VALUE_COLUMNS].mean().reset_index()
    valid_counts = (
        grouped[SUHI_COLUMNS]
        .count()
        .rename(columns={
            f"suhi_{slot}_mean": f"n_valid_months_{slot}"
            for slot in LST_SLOTS
        })
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
    return result


def aggregate_seasonal_suhi(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly values into fixed DJF/MAM/JJA/SON seasons."""
    seasonal = monthly.copy()
    seasonal["season"] = seasonal["month"].map(SEASON_BY_MONTH)
    seasonal["season_year"] = seasonal["year"]
    seasonal.loc[seasonal["month"] == 12, "season_year"] += 1

    result = _aggregate_monthly_values(
        panel=seasonal,
        group_columns=["ID_UC_G0", "season_year", "season"],
        expected_months=3,
        completeness_column="is_complete_season"
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


def aggregate_annual_suhi(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly values into calendar-year means."""
    return (
        _aggregate_monthly_values(
            panel=monthly,
            group_columns=["ID_UC_G0", "year"],
            expected_months=12,
            completeness_column="is_complete_year"
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
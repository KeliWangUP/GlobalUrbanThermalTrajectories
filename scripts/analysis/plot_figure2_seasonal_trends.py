#!/usr/bin/env python3
"""Plot paired city seasonal trends and urban-rural contrasts for Figure 2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


ROOT = Path(__file__).resolve().parents[2]
TREND_FILE = ROOT / "data/processed/gold/city_season_slot_trends_2003_2019_with_ghsl/city_season_slot_trends.parquet"
CONTRAST_FILE = ROOT / "data/processed/gold/city_seasonal_trend_contrasts_2003_2019_with_ghsl/city_seasonal_trend_contrasts.parquet"
SEASONAL_FILE = ROOT / "data/processed/suhi_seasonal.parquet"
OUTPUT_DIR = ROOT / "results/figures/figure2_seasonal_trends_Cfa"

# Input columns are kept here so schema adaptations are explicit.
FIELDS = {
    "city_id": "ID_UC_G0",
    "climate": "climate",
    "region": "GC_DEV_USR",
    "hemisphere": "hemisphere",
    "slot": "slot",
    "season": "season",
    "beta_urban": "beta_urban",
    "beta_rural": "beta_rural",
    "n_years": "n_valid_years",
    "warm_season": "summer_season",
    "cold_season": "winter_season",
    "beta_urban_warm": "beta_urban_summer",
    "beta_rural_warm": "beta_rural_summer",
    "beta_urban_cold": "beta_urban_winter",
    "beta_rural_cold": "beta_rural_winter",
    "n_years_warm": "n_valid_years_summer",
    "n_years_cold": "n_valid_years_winter",
}
GROUPS = (
    ("Cfa", "Europe"),
    # ("Cfa", "Europe"),
    # ("Cfa", "Eastern and South-Eastern Asia"),
    ("Cfa", "Eastern and South-Eastern Asia"),
    ("Cfa", "Northern America"),
)
BETAS = ("beta_urban_warm", "beta_rural_warm", "beta_urban_cold", "beta_rural_cold")
METRICS = BETAS + ("d_warm", "d_cold", "gamma_suhi")
SEED = 20261116
BOOTSTRAPS = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provenance(trend_file: Path, contrast_file: Path, seasonal_file: Path) -> tuple[int, int, int]:
    trend_manifest = json.loads((trend_file.parent / "manifest.json").read_text())
    contrast_manifest = json.loads((contrast_file.parent / "manifest.json").read_text())
    trend_hash, contrast_hash = sha256(trend_file), sha256(contrast_file)
    if (trend_manifest["output_sha256"] != trend_hash
            or contrast_manifest["output_sha256"] != contrast_hash
            or contrast_manifest["source_sha256"] != trend_hash
            or trend_manifest["source_sha256"] != sha256(seasonal_file)):
        raise ValueError("Input file hashes do not match their manifests and provenance.")
    if trend_manifest["beta_units"] != "degrees_Celsius_per_10_years" or trend_manifest["beta_scale_years"] != 10:
        raise ValueError("Trend manifest does not confirm degrees Celsius per decade.")
    if contrast_manifest["seasonal_direction"]["north"] != {"summer": "JJA", "winter": "DJF"}:
        raise ValueError("Northern seasonal definitions differ from JJA and DJF.")
    if contrast_manifest["seasonal_direction"]["south"] != {"summer": "DJF", "winter": "JJA"}:
        raise ValueError("Southern seasonal definitions differ from DJF and JJA.")
    years = trend_manifest["years"]
    print(f"Verified units: {trend_manifest['beta_units']}; season_year: {years['start']}-{years['end']}; minimum valid years: {years['minimum_valid_years']}.")
    print(f"Season definition: {trend_manifest['season_definition']}")
    print(f"Within each season, urban and rural share the QC/finite-observation mask: {trend_manifest['qc_policy']}")
    print("Gold tables alone lack year identities; the hash-verified seasonal panel will be used to recompute paired-year OLS trends.")
    return years["start"], years["end"], years["minimum_valid_years"]


def read_inputs(trend_file: Path, contrast_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Field mapping (logical -> input):")
    for logical, column in FIELDS.items():
        print(f"  {logical}: {column}")
    trends, contrasts = pd.read_parquet(trend_file), pd.read_parquet(contrast_file)
    trend_names = ("city_id", "climate", "region", "hemisphere", "slot", "season", "beta_urban", "beta_rural", "n_years")
    contrast_names = ("city_id", "climate", "region", "hemisphere", "slot", "warm_season", "cold_season", *BETAS, "n_years_warm", "n_years_cold")
    for label, frame, names, keys in (
        ("trends", trends, trend_names, ("city_id", "season", "slot")),
        ("contrasts", contrasts, contrast_names, ("city_id", "slot")),
    ):
        missing_columns = [FIELDS[n] for n in names if FIELDS[n] not in frame.columns]
        if missing_columns:
            raise ValueError(f"{label} missing required columns: {missing_columns}")
        print(f"{label}: {len(frame)} rows, {frame[FIELDS['city_id']].nunique()} distinct city IDs")
        for name in names:
            print(f"  missing {name}: {frame[FIELDS[name]].isna().sum()}")
        duplicate_count = int(frame.duplicated([FIELDS[k] for k in keys]).sum())
        print(f"  duplicate {'/'.join(keys)} records: {duplicate_count}")
        if duplicate_count or frame[[FIELDS[k] for k in keys]].isna().any().any():
            raise ValueError(f"{label} has duplicate or null record keys.")
    return (
        trends.rename(columns={FIELDS[n]: n for n in trend_names}),
        contrasts.rename(columns={FIELDS[n]: n for n in contrast_names}),
    )


def select_group(trends: pd.DataFrame, contrasts: pd.DataFrame, sensor: str,
                 group: tuple[str, str, str], period_length: int) -> pd.DataFrame:
    climate, region, time = group
    slot = f"{sensor}_{time}"
    def select(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[
            frame["climate"].eq(climate) & frame["region"].eq(region)
            & frame["hemisphere"].eq("north") & frame["slot"].eq(slot)
        ].copy()

    source = select(trends)
    source = source.loc[source["season"].isin(["JJA", "DJF"])]
    paired = select(contrasts)
    print(f"{sensor} {climate} | {region} | {time}: seasonal candidates={source['city_id'].nunique()} cities ({len(source)} rows), contrast candidates={len(paired)} cities")
    if not paired["warm_season"].eq("JJA").all() or not paired["cold_season"].eq("DJF").all():
        raise ValueError(f"Unexpected seasons in {sensor} {group}.")

    joined = None
    for season, suffix in (("JJA", "warm"), ("DJF", "cold")):
        rows = source.loc[source["season"].eq(season), ["city_id", "beta_urban", "beta_rural", "n_years"]].rename(
            columns={"beta_urban": f"beta_urban_{suffix}", "beta_rural": f"beta_rural_{suffix}", "n_years": f"n_years_{suffix}"}
        )
        joined = rows if joined is None else joined.merge(rows, on="city_id", how="outer", validate="one_to_one")
    assert joined is not None
    values = joined[list(BETAS)].apply(pd.to_numeric, errors="raise")
    valid = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    print(f"  missing/non-finite any of four betas: {int((~valid).sum())} cities")
    joined = joined.loc[valid].copy()
    matched = joined.merge(paired, on="city_id", how="outer", suffixes=("_source", "_contrast"), indicator=True, validate="one_to_one")
    print(f"  complete source/contrast matching: {matched['_merge'].value_counts().to_dict()}")
    if not matched["_merge"].eq("both").all():
        raise ValueError(f"Complete city sets differ between input tables for {sensor} {group}.")
    for field in (*BETAS, "n_years_warm", "n_years_cold"):
        if not np.allclose(matched[f"{field}_source"], matched[f"{field}_contrast"], rtol=1e-10, atol=1e-10, equal_nan=False):
            raise ValueError(f"Input tables disagree on {field} for {sensor} {group}.")
    if paired.empty:
        raise ValueError(f"No complete cities for {sensor} {group}.")
    paired = paired.sort_values("city_id").copy()
    if not np.isfinite(paired[list(BETAS)].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite contrast betas for {sensor} {group}.")
    paired = derive_contrasts(paired)
    full_years = paired["n_years_warm"].eq(period_length) & paired["n_years_cold"].eq(period_length)
    print(f"  Gold inputs: retained={len(paired)}; full-year matching inferred={int(full_years.sum())}; cross-season year sets unverified from Gold alone={int((~full_years).sum())}")
    return paired


def derive_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["d_warm"] = frame["beta_urban_warm"] - frame["beta_rural_warm"]
    frame["d_cold"] = frame["beta_urban_cold"] - frame["beta_rural_cold"]
    frame["gamma_suhi_derived"] = frame["d_warm"] - frame["d_cold"]
    gamma_urban = frame["beta_urban_warm"] - frame["beta_urban_cold"]
    gamma_rural = frame["beta_rural_warm"] - frame["beta_rural_cold"]
    if not np.allclose(frame["gamma_suhi_derived"], gamma_urban - gamma_rural, rtol=1e-10, atol=1e-10):
        raise ValueError("City-level gamma decomposition failed.")
    return frame


def paired_year_trends(seasonal_file: Path, contrasts: pd.DataFrame,
                       start: int, end: int, minimum: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-estimate slopes using common season years, then common day/night years."""
    candidates = contrasts.copy()
    valid = np.isfinite(candidates[list(BETAS)].to_numpy(dtype=float)).all(axis=1)
    valid &= candidates["climate"].notna() & candidates["region"].notna()
    print(f"Full-sample contrast rows excluded for missing grouping fields or non-finite betas: {int((~valid).sum())}")
    candidates = candidates.loc[valid].copy()
    expected_warm = candidates["hemisphere"].map({"north": "JJA", "south": "DJF"})
    expected_cold = candidates["hemisphere"].map({"north": "DJF", "south": "JJA"})
    if not (candidates["warm_season"].eq(expected_warm) & candidates["cold_season"].eq(expected_cold)).all():
        raise ValueError("Unexpected hemisphere or local seasons in contrast table.")
    metadata = candidates[["city_id", "slot", "hemisphere", "climate", "region"]]
    with duckdb.connect() as con:
        con.register("candidates", metadata)
        duplicates = con.execute(
            """SELECT COUNT(*) FROM (
                SELECT ID_UC_G0, season_year, season FROM read_parquet(?)
                GROUP BY 1, 2, 3 HAVING COUNT(*) > 1
            )""", [str(seasonal_file)]
        ).fetchone()[0]
        print(f"Seasonal panel duplicate city/year/season keys: {duplicates}")
        if duplicates:
            raise ValueError("Seasonal panel contains duplicate city/year/season keys.")
        values = ", ".join(
            f"('{sensor}_{time}', qc_pass_season_{sensor}_{time}, "
            f"urban_{sensor}_{time}_mean, rural_{sensor}_{time}_mean, suhi_{sensor}_{time}_mean)"
            for sensor in ("terra", "aqua") for time in ("day", "night")
        )
        con.execute(f"""
            CREATE TEMP TABLE season_pairs AS
            WITH observations AS (
                SELECT s.ID_UC_G0 AS city_id, s.season_year, s.season,
                       c.slot, c.hemisphere, v.urban, v.rural
                FROM read_parquet(?) s
                CROSS JOIN LATERAL (VALUES {values}) v(slot, qc, urban, rural, suhi)
                JOIN candidates c ON c.city_id = s.ID_UC_G0 AND c.slot = v.slot
                WHERE s.season_year BETWEEN ? AND ? AND s.season IN ('JJA', 'DJF')
                  AND v.qc IS TRUE AND isfinite(v.urban) AND isfinite(v.rural) AND isfinite(v.suhi)
            )
            SELECT w.city_id, w.slot, w.season_year,
                   w.urban AS beta_urban_warm, w.rural AS beta_rural_warm,
                   c.urban AS beta_urban_cold, c.rural AS beta_rural_cold
            FROM observations w JOIN observations c
              ON w.city_id = c.city_id AND w.slot = c.slot AND w.season_year = c.season_year
            WHERE w.season = CASE WHEN w.hemisphere = 'north' THEN 'JJA' ELSE 'DJF' END
              AND c.season = CASE WHEN c.hemisphere = 'north' THEN 'DJF' ELSE 'JJA' END
            """, [str(seasonal_file), start, end])

        slopes = ", ".join(f"10 * regr_slope({col}, season_year) AS {col}" for col in BETAS)
        def fit(relation: str) -> pd.DataFrame:
            fitted = con.execute(
                f"SELECT city_id, slot, COUNT(*) AS n_common_years, {slopes} "
                f"FROM {relation} GROUP BY city_id, slot HAVING COUNT(*) >= ? "
                "ORDER BY city_id, slot", [minimum]
            ).df()
            return derive_contrasts(fitted.merge(metadata, on=["city_id", "slot"], validate="one_to_one"))

        full = fit("season_pairs")
        con.execute("""
            CREATE TEMP TABLE day_night_pairs AS
            SELECT d.* FROM season_pairs d JOIN season_pairs n
              ON d.city_id = n.city_id AND d.season_year = n.season_year
             AND replace(d.slot, '_day', '_night') = n.slot
            WHERE ends_with(d.slot, '_day')
            UNION ALL
            SELECT n.* FROM season_pairs d JOIN season_pairs n
              ON d.city_id = n.city_id AND d.season_year = n.season_year
             AND replace(d.slot, '_day', '_night') = n.slot
            WHERE ends_with(d.slot, '_day')
        """)
        common = fit("day_night_pairs")
    print(f"All-sample paired-season city/slots: {len(full)} of {len(candidates)}; minimum common years: {minimum}.")
    print(f"Common day/night and season city/slots: {len(common)}.")
    return full, common


def summarize_full_sample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (slot, hemisphere, climate, region), group in frame.groupby(
        ["slot", "hemisphere", "climate", "region"], sort=True
    ):
        values = group["gamma_suhi_derived"].to_numpy(dtype=float)
        n = len(values)
        means = values[rng.integers(0, n, size=(BOOTSTRAPS, n))].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        reversal = int(((group["d_warm"] > 0) & (group["d_cold"] < 0)).sum())
        sensor, time = slot.split("_")
        rows.append({"sensor": sensor, "hemisphere": hemisphere, "climate": climate,
                     "region": region, "time": time, "n": n,
                     "mean_gamma_suhi": values.mean(), "ci_lower": low, "ci_upper": high,
                     "n_reversal": reversal, "share_reversal": reversal / n})
    return pd.DataFrame(rows)


def summarize_paired_group(day: pd.DataFrame, night: pd.DataFrame,
                           rng: np.random.Generator) -> tuple[list[pd.DataFrame], dict[str, float | int | str]]:
    columns = ["city_id", "n_common_years", *BETAS, "d_warm", "d_cold", "gamma_suhi_derived"]
    paired = day[columns].merge(
        night[columns],
        on="city_id", suffixes=("_day", "_night"), validate="one_to_one",
    )
    if len(paired) != len(day) or len(paired) != len(night) or paired.empty:
        raise ValueError("Day/night common-year city sets must be identical and non-empty.")
    if not paired["n_common_years_day"].eq(paired["n_common_years_night"]).all():
        raise ValueError("Day/night common-year counts differ.")
    metric_names = [*BETAS, "d_warm", "d_cold", "gamma_suhi_derived"]
    day_values = paired[[f"{name}_day" for name in metric_names]].to_numpy(dtype=float)
    night_values = paired[[f"{name}_night" for name in metric_names]].to_numpy(dtype=float)
    values = np.column_stack((day_values, night_values, day_values[:, 6] - night_values[:, 6]))
    n = len(values)
    means = values.mean(axis=0)
    for offset in (0, 7):
        if not np.isclose(means[offset + 6], means[offset + 4] - means[offset + 5], rtol=1e-10, atol=1e-10):
            raise ValueError("Group mean gamma identity failed.")
    replicates = values[rng.integers(0, n, size=(BOOTSTRAPS, n))].mean(axis=1)
    lower, upper = np.percentile(replicates, [2.5, 97.5], axis=0)
    summaries = [
        pd.DataFrame({"metric": METRICS, "n": n, "mean": means[offset:offset + 7],
                      "ci_lower": lower[offset:offset + 7], "ci_upper": upper[offset:offset + 7]})
        for offset in (0, 7)
    ]
    difference = {"n": n, "mean_gamma_day_minus_night": float(means[14]),
                  "ci_lower": float(lower[14]), "ci_upper": float(upper[14]),
                  "ci_relation_to_zero": "above_zero" if lower[14] > 0 else "below_zero" if upper[14] < 0 else "includes_zero",
                  "min_common_years": int(paired["n_common_years_day"].min()),
                  "max_common_years": int(paired["n_common_years_day"].max())}
    return summaries, difference


def summarize_group(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    values = frame[[*BETAS, "d_warm", "d_cold", "gamma_suhi_derived"]].to_numpy(dtype=float)
    n = len(values)
    if not n:
        raise ValueError("Cannot bootstrap an empty group.")
    means = values.mean(axis=0)
    if not np.isclose(means[6], means[4] - means[5], rtol=1e-10, atol=1e-10):
        raise ValueError("Group mean gamma identity failed.")
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
    replicates = values[indices].mean(axis=1)
    lower, upper = np.percentile(replicates, [2.5, 97.5], axis=0)
    return pd.DataFrame({"metric": METRICS, "n": n, "mean": means, "ci_lower": lower, "ci_upper": upper})


def axis_limits(summary: pd.DataFrame, metrics: tuple[str, ...]) -> tuple[float, float]:
    rows = summary.loc[summary["metric"].isin(metrics)]
    low = min(0.0, float(rows["ci_lower"].min()), float(rows["mean"].min()))
    high = max(0.0, float(rows["ci_upper"].max()), float(rows["mean"].max()))
    padding = max((high - low) * 0.09, 0.05)
    return low - padding, high + padding


def draw_figure(summary: pd.DataFrame, sensor: str, years: tuple[int, int, int],
                limits: tuple[tuple[float, float], tuple[float, float]], output: Path) -> None:
    fig = plt.figure(figsize=(17, 16), facecolor="white")
    data = summary.loc[summary["sensor"].eq(sensor)]
    positions = [0.8 + 4 * i for i in range(len(GROUPS))]
    for row_index, (time, bottom) in enumerate((("day", 0.555), ("night", 0.16))):
        left = fig.add_axes((0.245, bottom, 0.24, 0.33))
        right = fig.add_axes((0.75, bottom, 0.22, 0.33), sharey=left)
        for i, (climate, region) in enumerate(GROUPS):
            y = positions[i]
            rows = data.loc[(data["climate"] == climate) & (data["region"] == region) & (data["time"] == time)].set_index("metric")
            n = int(rows["n"].iloc[0])
            left.text(0.04, y - 0.75, f"{climate} · {region}  (n = {n})",
                      transform=blended_transform_factory(fig.transFigure, left.transData),
                      ha="left", va="center", fontsize=10, fontweight="bold", clip_on=False)
            for metric, yy, color in (
                ("beta_urban_warm", y - 0.13, "#D55E00"), ("beta_rural_warm", y + 0.13, "#0072B2"),
                ("beta_urban_cold", y + 0.87, "#D55E00"), ("beta_rural_cold", y + 1.13, "#0072B2"),
            ):
                point = rows.loc[metric]
                left.hlines(yy, point.ci_lower, point.ci_upper, color=color, linewidth=1.6)
                left.plot(point["mean"], yy, "o", color=color, markersize=6)
            for metric, yy, marker, color in (
                ("d_warm", y, "o", "#333333"), ("d_cold", y + 1, "o", "#333333"),
                ("gamma_suhi", y + 2, "D", "#7B3294"),
            ):
                point = rows.loc[metric]
                right.hlines(yy, point.ci_lower, point.ci_upper, color=color, linewidth=1.6)
                right.plot(point["mean"], yy, marker=marker, color=color, markersize=6)
        left.set_ylim(4 * len(GROUPS) - 0.4, -0.5)
        left.set_yticks([p + j for p in positions for j in (0, 1, 2)])
        left.set_yticklabels(["Warm season", "Cold season", ""] * len(GROUPS))
        right.tick_params(axis="y", labelleft=False)
        for y in positions:
            for offset, label in enumerate(("Warm season: urban − rural", "Cold season: urban − rural", "Warm − cold SUHI trend")):
                right.text(-0.03, y + offset, label, transform=blended_transform_factory(right.transAxes, right.transData),
                           ha="right", va="center", fontsize=9.5, clip_on=False)
        for axis, lim in zip((left, right), limits):
            axis.set_xlim(*lim)
            axis.axvline(0, color="#666666", linewidth=1, zorder=0)
            axis.grid(axis="x", color="#dddddd", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.tick_params(axis="y", length=0)
            for spine in ("top", "right", "left"):
                axis.spines[spine].set_visible(False)
        left.set_title(f"{'A' if row_index == 0 else 'C'}  {time.title()}time: LST trends", loc="left", fontweight="bold", fontsize=12, pad=12)
        right.set_title(f"{'B' if row_index == 0 else 'D'}  {time.title()}time: trend contrasts", loc="left", fontweight="bold", fontsize=12, pad=12)
        left.set_xlabel("LST trend β (°C decade⁻¹)")
        right.set_xlabel("Trend contrast (°C decade⁻¹)")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color="#D55E00", label="Urban"),
                        Line2D([], [], marker="o", linestyle="none", color="#0072B2", label="Rural")],
               loc="upper left", bbox_to_anchor=(0.245, 0.955), ncol=2, frameon=False)
    fig.suptitle(f"{sensor.title()} | Northern Hemisphere", x=0.245, y=0.985, ha="left", fontsize=14, fontweight="bold")
    caption = (
        f"Warm: June–August (JJA); cold: December–February (DJF). Season years {years[0]}–{years[1]}; December belongs to the next season year.\n"
        f"Within each sensor and group, day and night use the same cities and ≥{years[2]} common QC-valid season years across both seasons and times.\n"
        f"OLS trends are refitted on those years. Points are equal-city means; bars are 95% percentile intervals from {BOOTSTRAPS:,} paired city bootstrap resamples."
    )
    fig.text(0.04, 0.045, caption, ha="left", va="bottom", fontsize=9)
    for extension in ("png", "pdf"):
        path = output / f"figure2_{sensor}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(f"Output: {path}")
    plt.close(fig)


def draw_reversal_supplement(summary: pd.DataFrame, hemisphere: str, output: Path) -> None:
    data = summary.loc[summary["hemisphere"].eq(hemisphere)]
    climates = sorted(data["climate"].unique())
    regions = sorted(data["region"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(24, max(12, len(climates) * 0.52 + 6)), sharex=True, sharey=True)
    image = None
    for axis, (sensor, time) in zip(axes.flat, ((s, t) for s in ("terra", "aqua") for t in ("day", "night"))):
        part = data.loc[(data["sensor"] == sensor) & (data["time"] == time)]
        shares = part.pivot(index="climate", columns="region", values="share_reversal").reindex(index=climates, columns=regions)
        counts = part.pivot(index="climate", columns="region", values="n").reindex(index=climates, columns=regions)
        image = axis.imshow(shares.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(len(climates)):
            for j in range(len(regions)):
                count, share = counts.iloc[i, j], shares.iloc[i, j]
                if pd.notna(count):
                    axis.text(j, i, str(int(count)), ha="center", va="center",
                              color="white" if share < 0.55 else "black", fontsize=7)
        axis.set_title(f"{sensor.title()} · {time.title()}", fontweight="bold")
        axis.set_xticks(range(len(regions)), regions, rotation=40, ha="right", fontsize=8)
        axis.set_yticks(range(len(climates)), climates, fontsize=8)
        axis.tick_params(length=0, labelbottom=True, labelleft=True)
    assert image is not None
    fig.subplots_adjust(left=0.05, right=0.91, top=0.92, bottom=0.16, wspace=0.1, hspace=0.4)
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="Share with d_warm > 0 and d_cold < 0")
    fig.suptitle(f"Seasonal trend reversal | {hemisphere.title()}ern Hemisphere", fontsize=15, fontweight="bold")
    fig.text(0.05, 0.03, "Cell labels: number of cities; all non-empty groups shown. Local warm/cold seasons and paired-year OLS are used.", fontsize=10)
    for extension in ("png", "pdf"):
        path = output / f"figure2_reversal_supplement_{hemisphere}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(f"Output: {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trend-file", type=Path, default=TREND_FILE)
    parser.add_argument("--contrast-file", type=Path, default=CONTRAST_FILE)
    parser.add_argument("--seasonal-file", type=Path, default=SEASONAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    years = validate_provenance(args.trend_file, args.contrast_file, args.seasonal_file)
    trends, contrasts = read_inputs(args.trend_file, args.contrast_file)
    rng = np.random.default_rng(SEED)
    print(f"Paired city bootstrap: {BOOTSTRAPS} resamples, seed={SEED}.")
    full, common = paired_year_trends(args.seasonal_file, contrasts, *years)
    results = []
    differences = []
    for sensor in ("terra", "aqua"):
        for climate, region in GROUPS:
            original = {
                time: select_group(trends, contrasts, sensor, (climate, region, time), years[1] - years[0] + 1)
                for time in ("day", "night")
            }
            shared = set(original["day"]["city_id"]) & set(original["night"]["city_id"])
            selected = {
                time: common.loc[
                    common["slot"].eq(f"{sensor}_{time}") & common["hemisphere"].eq("north")
                    & common["climate"].eq(climate) & common["region"].eq(region)
                ].sort_values("city_id")
                for time in ("day", "night")
            }
            print(f"{sensor} {climate} | {region}: day n={len(original['day'])}; night n={len(original['night'])}; shared cities n={len(shared)}; common-year eligible n={len(selected['day'])}")
            if not set(selected["day"]["city_id"]).issubset(shared):
                raise ValueError("Common-year cities are not in the original shared city set.")
            summaries, difference = summarize_paired_group(selected["day"], selected["night"], rng)
            for time, stats in zip(("day", "night"), summaries):
                stats.insert(0, "time", time)
                stats.insert(0, "region", region)
                stats.insert(0, "climate", climate)
                stats.insert(0, "hemisphere", "north")
                stats.insert(0, "sensor", sensor)
                results.append(stats)
            differences.append({"sensor": sensor, "hemisphere": "north", "climate": climate,
                                "region": region, **difference})
            print(f"  Paired gamma day-minus-night: mean={difference['mean_gamma_day_minus_night']:.6g}, 95% CI=[{difference['ci_lower']:.6g}, {difference['ci_upper']:.6g}] ({difference['ci_relation_to_zero']}).")
    summary = pd.concat(results, ignore_index=True)
    limits = (axis_limits(summary, BETAS), axis_limits(summary, ("d_warm", "d_cold", "gamma_suhi")))
    print(f"Shared Terra/Aqua and day/night x limits: LST={limits[0]}, contrast={limits[1]}")
    full_summary = summarize_full_sample(full, rng)
    print(f"Full-sample groups: {len(full_summary)}; reversal numerator uses point estimates and denominator is all paired-season eligible cities.")
    print("Existing Figure 1 uses all nonempty groups, with no minimum city count; supplement follows the same rule.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sensor in ("terra", "aqua"):
        draw_figure(summary, sensor, years, limits, args.output_dir)
    for hemisphere in ("north", "south"):
        draw_reversal_supplement(full_summary, hemisphere, args.output_dir)
    for name, table in (
        ("figure2_summary.csv", summary),
        ("figure2_day_night_paired.csv", pd.DataFrame(differences)),
        ("figure2_full_sample_reversals.csv", full_summary),
    ):
        path = args.output_dir / name
        table.to_csv(path, index=False)
        print(f"Output: {path}")


if __name__ == "__main__":
    main()

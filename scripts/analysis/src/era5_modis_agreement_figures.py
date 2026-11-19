"""Compare paired seasonal air-temperature and urban/rural LST trends."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")
SLOT_LABELS = {
    "terra_day": "Terra Day", "terra_night": "Terra Night",
    "aqua_day": "Aqua Day", "aqua_night": "Aqua Night",
}
SERIES = ("era5", "urban", "rural")
SERIES_LABELS = {"era5": "ERA5 air", "urban": "Urban LST", "rural": "Rural LST"}
SERIES_COLORS = {"era5": "#0072B2", "urban": "#D55E00", "rural": "#009E73"}
BETAS = [f"beta_{series}_{season}" for series in SERIES for season in ("summer", "winter")]
GAMMAS = [f"gamma_{series}" for series in SERIES]
YEAR_START, YEAR_END, MIN_YEARS = 2003, 2019, 15
REQUIRED = {
    "ID_UC_G0", "slot", "hemisphere", "koppen_geiger_class", "GC_DEV_USR",
    "summer_season", "winter_season", "n_common_years", "common_years",
    "gamma_suhi", *BETAS, *GAMMAS,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise ValueError("Config must contain a paths object.")
    for key in ("contrast_file", "trend_manifest", "seasonal_source", "output_dir"):
        value = config["paths"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a non-empty path.")
        config["paths"][key] = (path.parent / Path(value).expanduser()).resolve()
    for key in ("reference_region", "comparison_region"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty region name.")
    if config["reference_region"] == config["comparison_region"]:
        raise ValueError("Reference and comparison regions must be distinct.")
    if type(config.get("dpi")) is not int or config["dpi"] < 1:
        raise ValueError("dpi must be a positive integer.")
    tolerance = config.get("tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    return config


def validate_provenance(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    for key in ("contrast_file", "trend_manifest", "seasonal_source"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"Missing {key}: {paths[key]}")
    manifest = json.loads(paths["trend_manifest"].read_text(encoding="utf-8"))
    years = manifest.get("years", {})
    if (years.get("start"), years.get("end"), years.get("minimum_valid_years")) != (YEAR_START, YEAR_END, MIN_YEARS):
        raise ValueError("Trend manifest must specify 2003–2019 and at least 15 paired years.")
    if manifest.get("beta_units") != "degrees_Celsius_per_10_years" or manifest.get("beta_scale_years") != 10:
        raise ValueError("Trend manifest must specify degrees Celsius per decade.")
    if _sha256(paths["contrast_file"]) != manifest["outputs"]["contrast"]["sha256"]:
        raise ValueError("Contrast file does not match the trend manifest checksum.")
    if _sha256(paths["seasonal_source"]) != manifest["source_sha256"]:
        raise ValueError("Seasonal source does not match the trend manifest checksum.")
    print("[INPUT] Verified 2003–2019, 15-year threshold, °C/decade and source checksums.")
    return manifest


def validate_and_select(frame: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    missing_columns = REQUIRED - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Contrast table missing columns: {sorted(missing_columns)}")
    print(f"[INPUT] Contrast rows: {len(frame):,}; cities: {frame['ID_UC_G0'].nunique():,}")
    duplicates = int(frame.duplicated(["ID_UC_G0", "slot"]).sum())
    print(f"[CHECK] Duplicate city/slot rows: {duplicates:,}")
    if duplicates or frame["ID_UC_G0"].isna().any():
        raise ValueError("Contrast table contains duplicate or missing city/slot keys.")
    if frame["slot"].isna().any() or not frame["slot"].isin(SLOTS).all():
        raise ValueError("Contrast table contains unsupported or missing slots.")
    if frame["hemisphere"].isna().any() or not frame["hemisphere"].isin(["north", "south"]).all():
        raise ValueError("Contrast table contains unsupported or missing hemispheres.")
    bad_seasons = (
        ((frame["hemisphere"] == "north") & ((frame["summer_season"] != "JJA") | (frame["winter_season"] != "DJF")))
        | ((frame["hemisphere"] == "south") & ((frame["summer_season"] != "DJF") | (frame["winter_season"] != "JJA")))
    )
    if bad_seasons.any():
        raise ValueError("Contrast seasons do not match hemisphere definitions.")

    numeric = frame[BETAS + GAMMAS + ["gamma_suhi"]].apply(pd.to_numeric, errors="raise")
    missing_counts = numeric.isna().sum().to_dict()
    print(f"[CHECK] Missing beta/gamma values by field: {missing_counts}")
    nonfinite = ~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    print(f"[FILTER] Excluded non-finite beta/gamma rows: {int(nonfinite.sum()):,}")
    frame = frame.loc[~nonfinite].copy()
    frame.loc[:, numeric.columns] = numeric.loc[~nonfinite]
    invalid_years = []
    for count, values in zip(frame["n_common_years"], frame["common_years"]):
        if values is None:
            invalid_years.append(True)
            continue
        years = list(values)
        invalid_years.append(
            pd.isna(count) or len(years) != count or len(years) < MIN_YEARS
            or any(not isinstance(year, (int, np.integer)) or not YEAR_START <= year <= YEAR_END for year in years)
            or years != sorted(set(years))
        )
    print(f"[CHECK] Invalid common-year lists: {sum(invalid_years):,}")
    if any(invalid_years):
        raise ValueError("Contrast rows have invalid common-year lists.")
    errors = {
        f"gamma_{series}": float((frame[f"gamma_{series}"] - (frame[f"beta_{series}_summer"] - frame[f"beta_{series}_winter"])).abs().max())
        for series in SERIES
    }
    errors["gamma_suhi"] = float((frame["gamma_suhi"] - (frame["gamma_urban"] - frame["gamma_rural"])).abs().max())
    print(f"[CHECK] Maximum decomposition errors: {errors}")
    if not errors or any(not math.isfinite(error) or error > tolerance for error in errors.values()):
        raise ValueError("Trend contrast decomposition failed.")
    climate = frame["koppen_geiger_class"].astype("string").str.strip()
    missing_climate = int(climate.isna().sum())
    selected = frame.loc[(frame["hemisphere"] == "north") & climate.str.startswith("C", na=False)].copy()
    print(f"[FILTER] Missing climate rows: {missing_climate:,}; northern Köppen C rows: {len(selected):,}")
    if selected.empty:
        raise ValueError("No northern Köppen C city-slot rows remain.")
    selected["modis_gamma_mean"] = (selected["gamma_urban"] + selected["gamma_rural"]) / 2
    selected["gamma_urban_minus_rural"] = selected["gamma_urban"] - selected["gamma_rural"]
    return selected


def validate_regions(frame: pd.DataFrame) -> pd.DataFrame:
    if "GC_DEV_USR" not in frame.columns:
        raise ValueError("Contrast table missing region field GC_DEV_USR.")
    region = frame["GC_DEV_USR"]
    if not region.dropna().map(lambda value: isinstance(value, str)).all():
        raise ValueError("Contrast region values must be strings.")
    region = region.astype("string").str.strip()
    missing = int((region.isna() | region.eq("")).sum())
    print(f"[CHECK] Missing or empty contrast regions in selected rows: {missing:,}")
    if missing:
        raise ValueError("Selected contrast rows contain missing or empty regions.")
    result = frame.copy()
    result["GC_DEV_USR"] = region
    inconsistent = int((result.groupby("ID_UC_G0")["GC_DEV_USR"].nunique() > 1).sum())
    print(f"[CHECK] Cities with inconsistent contrast regions across slots: {inconsistent:,}")
    if inconsistent:
        raise ValueError("Selected contrast rows have inconsistent city regions across slots.")
    return result


def verify_common_year_fits(frame: pd.DataFrame, source: Path, tolerance: float) -> None:
    """Reconstruct common-year observations and verify all six saved slopes."""
    candidates = frame[["ID_UC_G0", "slot"]].copy()
    values = ", ".join(
        f"('{slot}', qc_pass_season_{slot}, urban_{slot}_mean, rural_{slot}_mean, suhi_{slot}_mean)"
        for slot in SLOTS
    )
    with duckdb.connect() as con:
        con.register("candidates", candidates)
        fits = con.execute(
            f"""
            WITH obs AS (
                SELECT s.ID_UC_G0, s.season_year, s.season, v.slot,
                       v.urban, v.rural, s.era5_temperature_2m_c AS era5
                FROM read_parquet(?) s
                CROSS JOIN LATERAL (VALUES {values}) v(slot, qc, urban, rural, suhi)
                JOIN candidates c ON s.ID_UC_G0=c.ID_UC_G0 AND v.slot=c.slot
                WHERE s.season_year BETWEEN {YEAR_START} AND {YEAR_END}
                  AND s.season IN ('JJA', 'DJF') AND v.qc IS TRUE
                  AND s.era5_qc_pass IS TRUE
                  AND isfinite(v.urban) AND isfinite(v.rural) AND isfinite(v.suhi)
                  AND isfinite(s.era5_temperature_2m_c)
            ),
            pairs AS (
                SELECT w.ID_UC_G0, w.slot, w.season_year,
                       w.urban urban_summer, c.urban urban_winter,
                       w.rural rural_summer, c.rural rural_winter,
                       w.era5 era5_summer, c.era5 era5_winter
                FROM obs w JOIN obs c
                  ON w.ID_UC_G0=c.ID_UC_G0 AND w.slot=c.slot AND w.season_year=c.season_year
                WHERE w.season='JJA' AND c.season='DJF'
            )
            SELECT ID_UC_G0, slot, list(season_year ORDER BY season_year) AS verified_years,
                   {", ".join(f"10 * regr_slope({series}_{season}, season_year) AS verified_{series}_{season}" for series in SERIES for season in ("summer", "winter"))}
            FROM pairs GROUP BY ID_UC_G0, slot
            """,
            [str(source)],
        ).df()
    checked = frame.merge(fits, on=["ID_UC_G0", "slot"], how="left", validate="one_to_one")
    bad_years = sum(
        years is None or list(years) != list(expected)
        for years, expected in zip(checked["verified_years"], checked["common_years"])
    )
    error = 0.0
    for series in SERIES:
        for season in ("summer", "winter"):
            differences = (checked[f"verified_{series}_{season}"] - checked[f"beta_{series}_{season}"]).abs()
            if differences.isna().any():
                raise ValueError("Unable to independently refit all requested beta values.")
            error = max(error, float(differences.max()))
    print(f"[CHECK] Common-year list mismatches: {bad_years}; six-beta max refit error: {error:.3g}")
    if bad_years or error > tolerance:
        raise ValueError("Saved common years or six trend slopes do not match seasonal data.")


def summaries(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    metrics = GAMMAS + BETAS + ["gamma_urban_minus_rural"]
    records = []
    for (slot, region), group in frame.groupby(["slot", "GC_DEV_USR"], sort=True):
        record: dict[str, Any] = {"cohort": cohort, "slot": slot, "region": region, "n": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            record[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            record[f"{metric}_mean"] = float(np.mean(values))
        for series, name in (("modis_gamma_mean", "modis_mean"), ("gamma_urban", "urban"), ("gamma_rural", "rural")):
            air = group["gamma_era5"].to_numpy(dtype=float)
            other = group[series].to_numpy(dtype=float)
            record[f"pearson_air_vs_{name}"] = (
                float(np.corrcoef(air, other)[0, 1])
                if len(group) >= 2 and np.ptp(air) > 0 and np.ptp(other) > 0 else None
            )
        records.append(record)
    return pd.DataFrame(records)


def regional_differences(summary: pd.DataFrame, reference: str, comparison: str) -> pd.DataFrame:
    rows = []
    for (cohort, slot), group in summary.groupby(["cohort", "slot"], sort=True):
        ref = group.loc[group["region"] == reference]
        comp = group.loc[group["region"] == comparison]
        if len(ref) != 1 or len(comp) != 1:
            raise ValueError(f"Both comparison regions must have samples for {cohort}/{slot}.")
        ref_row, comp_row = ref.iloc[0], comp.iloc[0]
        for metric in GAMMAS + BETAS + ["gamma_urban_minus_rural"]:
            rows.append({
                "cohort": cohort, "slot": slot, "metric": metric,
                "reference_region": reference, "comparison_region": comparison,
                "n_reference": int(ref_row["n"]), "n_comparison": int(comp_row["n"]),
                "reference_median": ref_row[f"{metric}_median"],
                "comparison_median": comp_row[f"{metric}_median"],
                "median_difference_reference_minus_comparison": ref_row[f"{metric}_median"] - comp_row[f"{metric}_median"],
                "mean_difference_reference_minus_comparison": ref_row[f"{metric}_mean"] - comp_row[f"{metric}_mean"],
            })
    return pd.DataFrame(rows)


def _save(figure: plt.Figure, base: Path, dpi: int) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_scatter(frame: pd.DataFrame, label: str, colors: dict[str, Any], highlight: set[str]) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    extent = max(float(frame["gamma_era5"].abs().max()), float(frame["modis_gamma_mean"].abs().max()), 0.1) * 1.05
    categories = sorted(frame["GC_DEV_USR"].unique())
    for axis, slot in zip(axes.flat, SLOTS):
        part = frame.loc[frame["slot"] == slot]
        for region in sorted(categories, key=lambda x: x in highlight):
            values = part.loc[part["GC_DEV_USR"] == region]
            axis.scatter(values["gamma_era5"], values["modis_gamma_mean"],
                         color=colors[region], s=18 if region in highlight else 13,
                         alpha=0.75 if region in highlight else 0.30, linewidths=0)
        axis.axhline(0, color="0.5", linewidth=0.7)
        axis.axvline(0, color="0.5", linewidth=0.7)
        axis.plot([-extent, extent], [-extent, extent], color="0.75", linewidth=1)
        axis.set(xlim=(-extent, extent), ylim=(-extent, extent), title=f"{SLOT_LABELS[slot]} (N={len(part):,})")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    figure.supxlabel("ERA5 γair (°C/decade)", y=0.16)
    figure.supylabel("MODIS M = (γU + γR) / 2 (°C/decade)")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[region], label=region, markersize=6) for region in categories]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.035))
    figure.text(0.5, 0.01, "Paired season-years, not necessarily the same valid months; slots analyzed separately.",
                ha="center", fontsize=9)
    figure.suptitle(f"{label} — agreement with regional air-temperature trends\nGray line: numerical equality, not a physical identity")
    figure.tight_layout(rect=(0.03, 0.23, 1, 0.91))
    return figure


def _draw_region_intervals(axis: plt.Axes, table: pd.DataFrame, metrics: list[str], regions: list[str]) -> None:
    offsets = (-0.23, 0, 0.23)
    for series, metric, offset in zip(SERIES, metrics, offsets):
        for position, region in enumerate(regions):
            row = table.loc[table["region"] == region]
            if row.empty:
                continue
            row = row.iloc[0]
            median, low, high = (row[f"{metric}_{stat}"] for stat in ("median", "q25", "q75"))
            axis.errorbar(median, position + offset, xerr=[[median - low], [high - median]],
                          fmt="o", markersize=4, capsize=2, linewidth=1.1,
                          color=SERIES_COLORS[series])
    axis.set_yticks(range(len(regions)), regions)
    axis.invert_yaxis()
    axis.axvline(0, color="0.45", linewidth=0.8)
    axis.grid(axis="x", alpha=0.2)


def _legend() -> list[Line2D]:
    return [Line2D([0], [0], color=SERIES_COLORS[name], marker="o", linestyle="-",
                   label=SERIES_LABELS[name]) for name in SERIES]


def plot_gamma(summary: pd.DataFrame, label: str) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    regions = sorted(summary["region"].unique())
    for axis, slot in zip(axes.flat, SLOTS):
        part = summary.loc[summary["slot"] == slot]
        _draw_region_intervals(axis, part, GAMMAS, regions)
        axis.set_yticklabels([f"{region} (N={int(part.loc[part['region'] == region, 'n'].iloc[0])})"
                              if (part["region"] == region).any() else f"{region} (N=0)"
                              for region in regions])
        axis.set_title(SLOT_LABELS[slot])
    figure.supxlabel("Summer − winter trend γ (°C/decade)", y=0.11)
    figure.legend(handles=_legend(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.035))
    figure.text(0.5, 0.01, "Paired season-years, not necessarily the same valid months; slots analyzed separately.",
                ha="center", fontsize=9)
    figure.suptitle(f"{label} — agreement with regional air-temperature trends\nCity medians and interquartile ranges; same cities for all three series")
    figure.tight_layout(rect=(0, 0.16, 1, 0.93))
    return figure


def plot_betas(summary: pd.DataFrame, label: str, slot: str) -> plt.Figure:
    part = summary.loc[summary["slot"] == slot]
    regions = sorted(part["region"].unique())
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for axis, season, display in zip(axes, ("summer", "winter"), ("Warm season", "Cold season")):
        _draw_region_intervals(axis, part, [f"beta_{series}_{season}" for series in SERIES], regions)
        axis.set_yticklabels([f"{region} (N={int(part.loc[part['region'] == region, 'n'].iloc[0])})" for region in regions])
        axis.set_title(display)
    figure.supxlabel("Seasonal trend β (°C/decade)", y=0.11)
    figure.legend(handles=_legend(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.035))
    figure.text(0.5, 0.01, "Paired season-years, not necessarily the same valid months; slots analyzed separately.",
                ha="center", fontsize=9)
    figure.suptitle(f"{label} — {SLOT_LABELS[slot]}\nagreement with regional air-temperature trends; city medians and interquartile ranges")
    figure.tight_layout(rect=(0, 0.16, 1, 0.91))
    return figure


def _difference_report(differences: pd.DataFrame) -> str:
    lines = [
        "Regional differences: reference minus comparison (degrees Celsius per decade).",
        "Medians of beta differences do not generally decompose median gamma differences.",
        "Mean beta differences do decompose mean gamma differences when the same cities are used.",
        "Pearson correlations in the summary describe association, not agreement or causality.",
        "Slots are separate samples: these results do not quantify day/night differences.",
        "",
    ]
    for (cohort, slot), group in differences.groupby(["cohort", "slot"], sort=True):
        region_a = group["reference_region"].iloc[0]
        region_b = group["comparison_region"].iloc[0]
        lines.append(f"{cohort} | {SLOT_LABELS[slot]} | {region_a} minus {region_b} | N={group['n_reference'].iloc[0]} vs {group['n_comparison'].iloc[0]}")
        values = group.set_index("metric")
        for series in SERIES:
            gamma = values.loc[f"gamma_{series}"]
            warm = values.loc[f"beta_{series}_summer"]
            cold = values.loc[f"beta_{series}_winter"]
            lines.append(
                f"  {SERIES_LABELS[series]}: median delta gamma={gamma['median_difference_reference_minus_comparison']:+.3f}; "
                f"median delta warm beta={warm['median_difference_reference_minus_comparison']:+.3f}; "
                f"median delta cold beta={cold['median_difference_reference_minus_comparison']:+.3f}; "
                f"mean delta gamma={gamma['mean_difference_reference_minus_comparison']:+.3f} "
                f"= mean delta warm beta {warm['mean_difference_reference_minus_comparison']:+.3f} "
                f"- mean delta cold beta {cold['mean_difference_reference_minus_comparison']:+.3f}"
            )
        lines.append("")
    return "\n".join(lines)


def build_figures(config_path: Path) -> None:
    config = load_config(config_path)
    paths = config["paths"]
    validate_provenance(config)
    source = pd.read_parquet(paths["contrast_file"])
    frame = validate_and_select(source, config["tolerance"])
    frame = validate_regions(frame)
    verify_common_year_fits(frame, paths["seasonal_source"], config["tolerance"])
    categories = sorted(frame["GC_DEV_USR"].unique())
    colors = dict(zip(categories, plt.get_cmap("tab20")(np.linspace(0, 1, len(categories)))))
    highlight = {config["reference_region"], config["comparison_region"]}
    cohorts = {
        "koppen_C_north": ("Northern Hemisphere Köppen C", frame),
        "Cfa_north": ("Northern Hemisphere Cfa", frame.loc[frame["koppen_geiger_class"].eq("Cfa")].copy()),
    }
    for cohort, (_, subset) in cohorts.items():
        if subset.empty:
            raise ValueError(f"No records in {cohort}.")
        print(f"[SAMPLE] {cohort}: {len(subset):,} city-slot records; {subset['ID_UC_G0'].nunique():,} distinct cities")
        for (slot, region), group in subset.groupby(["slot", "GC_DEV_USR"], sort=True):
            print(f"  {SLOT_LABELS[slot]} | {region}: N={len(group)}")
    combined_summary = pd.concat([summaries(subset, name) for name, (_, subset) in cohorts.items()], ignore_index=True)
    differences = regional_differences(combined_summary, config["reference_region"], config["comparison_region"])
    output = paths["output_dir"]
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    with tempfile.TemporaryDirectory(prefix=f".{output.name}_staging_", dir=output.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        for name, (label, subset) in cohorts.items():
            _save(plot_scatter(subset, label, colors, highlight), stage / f"figure1_scatter_{name}", config["dpi"])
            summary = combined_summary.loc[combined_summary["cohort"] == name]
            _save(plot_gamma(summary, label), stage / f"figure2_gamma_{name}", config["dpi"])
            for slot in SLOTS:
                _save(plot_betas(summary, label, slot), stage / f"figure3_beta_{name}_{slot}", config["dpi"])
            city_data = subset.copy()
            city_data["common_years"] = city_data["common_years"].map(lambda years: ";".join(str(year) for year in years))
            city_data.to_csv(stage / f"city_paired_values_{name}.csv", index=False, encoding="utf-8")
        combined_summary.to_csv(stage / "regional_summary.csv", index=False, encoding="utf-8")
        differences.to_csv(stage / "europe_east_asia_differences.csv", index=False, encoding="utf-8")
        (stage / "regional_difference_report.txt").write_text(_difference_report(differences), encoding="utf-8")
        manifest = {
            "contrast_file": str(paths["contrast_file"]), "contrast_sha256": _sha256(paths["contrast_file"]),
            "seasonal_source_sha256": _sha256(paths["seasonal_source"]),
            "region_source": "contrast_file.GC_DEV_USR",
            "period": "2003–2019", "minimum_common_years": 15,
            "beta_units": "degrees_Celsius_per_decade",
            "common_year_fits_verified": True,
            "contrast": f"{config['reference_region']} minus {config['comparison_region']}",
            "region_colors": {region: matplotlib.colors.to_hex(color) for region, color in colors.items()},
            "cohort_city_slot_rows": {name: len(subset) for name, (_, subset) in cohorts.items()},
            "day_night_comparison": "Not computed; each slot uses its own paired sample.",
            "median_decomposition": "Median beta differences need not add to median gamma differences.",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output directory appeared during build: {output}")
        stage.rename(output)
    print(_difference_report(differences))
    print(f"[OUTPUT] {output}")

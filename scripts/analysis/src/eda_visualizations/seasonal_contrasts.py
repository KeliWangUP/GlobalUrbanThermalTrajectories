import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda_seasonal_contrast import (
    DATASETS,
    METRICS,
    SENSORS,
    SeasonalContrastResults
)


LOGGER = logging.getLogger("diurnal_eda")
DATASET_COLORS = {
    "urban": "#d55e00",
    "rural": "#0072b2",
    "suhi": "#009e73"
}
METRIC_TITLES = {
    "D": "Daytime",
    "N": "Nighttime",
    "A": "Day − night"
}


def _climate_order(climate_groups: tuple[str, ...]) -> list[str]:
    return ["All cities", *climate_groups, "Unclassified"]


def _plot_median_trends(
    summary: pd.DataFrame,
    output_path: Path,
    climate_groups: tuple[str, ...],
    confidence_level: float,
    dpi: int
) -> None:
    climate_order = _climate_order(climate_groups)
    offsets = {"urban": -0.22, "rural": 0.0, "suhi": 0.22}
    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(19, 10),
        sharey=True
    )
    for row, sensor in enumerate(SENSORS):
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            subset = summary.loc[
                summary["sensor"].eq(sensor)
                & summary["metric"].eq(metric)
            ]
            for dataset in DATASETS:
                data = (
                    subset.loc[subset["dataset"].eq(dataset)]
                    .set_index("climate_group")
                    .reindex(climate_order)
                )
                medians = data["median_gamma_per_decade"].to_numpy(
                    dtype="float64"
                )
                lower = data["ci_lower"].to_numpy(dtype="float64")
                upper = data["ci_upper"].to_numpy(dtype="float64")
                valid = np.isfinite(medians) & np.isfinite(lower) & np.isfinite(upper)
                positions = np.arange(len(climate_order), dtype="float64")
                positions += offsets[dataset]
                axis.errorbar(
                    positions[valid],
                    medians[valid],
                    yerr=np.vstack((
                        np.maximum(0, medians[valid] - lower[valid]),
                        np.maximum(0, upper[valid] - medians[valid])
                    )),
                    fmt="o",
                    markersize=4,
                    capsize=2,
                    linewidth=1,
                    color=DATASET_COLORS[dataset],
                    label=dataset.title()
                )
            axis.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
            axis.set_xticks(
                np.arange(len(climate_order)),
                climate_order,
                rotation=40,
                ha="right"
            )
            axis.set_title(f"{sensor.title()} — {METRIC_TITLES[metric]}")
            axis.grid(axis="y", alpha=0.2, linewidth=0.5)
            if column == 0:
                axis.set_ylabel("Median γ (°C per decade)")
    figure.suptitle(
        "Warm − cold seasonal-contrast trends "
        f"({confidence_level:.0%} city-bootstrap CI)"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False
    )
    figure.tight_layout(rect=(0.02, 0.03, 0.99, 0.88))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_positive_shares(
    summary: pd.DataFrame,
    output_path: Path,
    climate_groups: tuple[str, ...],
    dpi: int
) -> None:
    climate_order = _climate_order(climate_groups)
    offsets = {"urban": -0.22, "rural": 0.0, "suhi": 0.22}
    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(19, 10),
        sharey=True
    )
    for row, sensor in enumerate(SENSORS):
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            subset = summary.loc[
                summary["sensor"].eq(sensor)
                & summary["metric"].eq(metric)
            ]
            for dataset in DATASETS:
                data = (
                    subset.loc[subset["dataset"].eq(dataset)]
                    .set_index("climate_group")
                    .reindex(climate_order)
                )
                shares = data["positive_gamma_pct"].to_numpy(
                    dtype="float64"
                )
                valid = np.isfinite(shares)
                positions = np.arange(len(climate_order), dtype="float64")
                positions += offsets[dataset]
                axis.scatter(
                    positions[valid],
                    shares[valid],
                    s=24,
                    color=DATASET_COLORS[dataset],
                    label=dataset.title(),
                    zorder=2
                )
            axis.axhline(50, color="0.35", linewidth=0.8, linestyle="--")
            axis.set_ylim(0, 100)
            axis.set_xticks(
                np.arange(len(climate_order)),
                climate_order,
                rotation=40,
                ha="right"
            )
            axis.set_title(f"{sensor.title()} — {METRIC_TITLES[metric]}")
            axis.grid(axis="y", alpha=0.2, linewidth=0.5)
            if column == 0:
                axis.set_ylabel("Cities with γ > 0 (%)")
    figure.suptitle(
        "Share of cities with increasing warm − cold seasonal contrast"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False
    )
    figure.tight_layout(rect=(0.02, 0.03, 0.99, 0.88))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def create_seasonal_contrast_outputs(
    results: SeasonalContrastResults,
    output_dir: Path,
    start_year: int,
    end_year: int,
    climate_groups: tuple[str, ...],
    confidence_level: float,
    parquet_compression: str,
    dpi: int
) -> list[Path]:
    """Write analysis tables and plots for warm/cold seasonal contrasts."""
    output_dir = output_dir / "warm_cold_seasonal_contrasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    period = f"{start_year}-{end_year}"
    outputs = []

    labels_path = output_dir / f"city_month_season_labels_{period}.parquet"
    results.city_month_labels.to_parquet(
        labels_path,
        engine="pyarrow",
        compression=parquet_compression,
        index=False
    )
    outputs.append(labels_path)

    tables = (
        (
            f"city_year_season_values_{period}.csv",
            results.city_year_seasons
        ),
        (
            f"city_year_warm_minus_cold_contrasts_{period}.csv",
            results.city_year_contrasts
        ),
        (
            f"city_warm_cold_contrast_trends_{period}.csv",
            results.city_trends
        ),
        (
            f"climate_warm_cold_contrast_summary_{period}.csv",
            results.climate_summary
        )
    )
    for file_name, frame in tables:
        path = output_dir / file_name
        frame.to_csv(path, index=False)
        outputs.append(path)

    median_plot = output_dir / f"median_gamma_by_climate_{period}.png"
    _plot_median_trends(
        summary=results.climate_summary,
        output_path=median_plot,
        climate_groups=climate_groups,
        confidence_level=confidence_level,
        dpi=dpi
    )
    outputs.append(median_plot)

    share_plot = output_dir / f"positive_city_shares_{period}.png"
    _plot_positive_shares(
        summary=results.climate_summary,
        output_path=share_plot,
        climate_groups=climate_groups,
        dpi=dpi
    )
    outputs.append(share_plot)

    LOGGER.info(
        "[SEASONAL CONTRAST] Wrote %s city-month labels, %s city-year-season "
        "rows, %s city-year contrasts, and %s city-sensor trends.",
        len(results.city_month_labels),
        len(results.city_year_seasons),
        len(results.city_year_contrasts),
        len(results.city_trends)
    )
    for path in outputs:
        LOGGER.info("Seasonal contrast output: %s", path)
    return outputs

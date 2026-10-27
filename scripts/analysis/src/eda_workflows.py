import logging

import pandas as pd

from src.config import AnalysisConfig, load_analysis_config
from src.eda_logging import configure_eda_logging
from src.eda_visualizations.calendar_month import (
    create_calendar_month_figures
)
from src.eda_visualizations.city_slopes import create_city_slope_figures
from src.eda_visualizations.climate_month_trends import (
    create_climate_month_trend_heatmaps
)
from src.eda_visualizations.local_season import (
    create_local_season_month_figures,
    create_local_season_year_figures
)
from src.eda_visualizations.year_panels import (
    create_calendar_year_figures
)
from src.eda_visualizations.yearly_trends import (
    create_yearly_trend_figures
)
from src.eda_seasonal_contrast import analyze_seasonal_contrasts
from src.eda_visualizations.seasonal_contrasts import (
    create_seasonal_contrast_outputs
)


LOGGER = logging.getLogger("diurnal_eda")


def _load_panel(
    config: AnalysisConfig,
    require_centroid: bool,
    require_climate_macro_group: bool = False
) -> pd.DataFrame:
    if not config.monthly_suhi_file.is_file():
        raise FileNotFoundError(
            "Monthly SUHI master table not found: "
            f"{config.monthly_suhi_file}. Run 02_build_monthly_suhi.py first."
        )
    LOGGER.info("Input: %s", config.monthly_suhi_file)
    panel = pd.read_parquet(config.monthly_suhi_file)
    if require_centroid and "centroid_lat" not in panel.columns:
        raise ValueError(
            "Monthly SUHI panel is missing centroid_lat. Run "
            "04_merge_city_metadata.py or rebuild it with "
            "02_build_monthly_suhi.py."
        )
    if (
        require_climate_macro_group
        and "koppen_geiger_macro_group" not in panel.columns
    ):
        raise ValueError(
            "Monthly SUHI panel is missing koppen_geiger_macro_group. "
            "Run 04_merge_city_metadata.py to refresh city metadata."
        )
    return panel


def run_eda_workflow(workflow_name: str) -> None:
    config = load_analysis_config()
    logger, _ = configure_eda_logging(config.eda_log_dir)
    logger.info(
        "Starting EDA workflow '%s' for %s-%s (%s aggregation).",
        workflow_name,
        config.eda.start_year,
        config.eda.end_year,
        config.eda.city_month_aggregation
    )

    try:
        if workflow_name == "calendar_months":
            workflows = ("calendar_months",)
        elif workflow_name == "yearly_panels":
            workflows = ("yearly_panels",)
        elif workflow_name == "local_season":
            workflows = ("local_season",)
        elif workflow_name == "yearly_trends":
            workflows = ("yearly_trends",)
        elif workflow_name == "city_slopes":
            workflows = ("city_slopes",)
        elif workflow_name == "climate_month_trends":
            workflows = ("climate_month_trends",)
        elif workflow_name == "seasonal_contrasts":
            workflows = ("seasonal_contrasts",)
        elif workflow_name == "all":
            workflows = (
                "calendar_months",
                "yearly_panels",
                "local_season",
                "yearly_trends",
                "city_slopes",
                "climate_month_trends",
                "seasonal_contrasts"
            )
        else:
            raise ValueError(f"Unknown EDA workflow: {workflow_name}")

        panel = _load_panel(
            config,
            require_centroid=(
                "local_season" in workflows
                or "yearly_trends" in workflows
                or "city_slopes" in workflows
                or "climate_month_trends" in workflows
                or "seasonal_contrasts" in workflows
            ),
            require_climate_macro_group=(
                "city_slopes" in workflows
                or "climate_month_trends" in workflows
                or "seasonal_contrasts" in workflows
            )
        )
        for workflow in workflows:
            if workflow == "calendar_months":
                outputs = create_calendar_month_figures(
                    panel,
                    config.eda_figure_dir,
                    config.eda.start_year,
                    config.eda.end_year,
                    config.eda.city_month_aggregation,
                    config.eda.figure_dpi
                )
            elif workflow == "yearly_panels":
                outputs = create_calendar_year_figures(
                    panel,
                    config.eda_figure_dir,
                    config.eda.start_year,
                    config.eda.end_year,
                    config.eda.figure_dpi
                )
            elif workflow == "local_season":
                outputs = create_local_season_month_figures(
                    panel,
                    config.eda_figure_dir,
                    config.eda.start_year,
                    config.eda.end_year,
                    config.eda.city_month_aggregation,
                    config.eda.figure_dpi,
                    config.eda.equatorial_latitude_band_degrees
                )
                outputs.extend(create_local_season_year_figures(
                    panel,
                    config.eda_figure_dir,
                    config.eda.start_year,
                    config.eda.end_year,
                    config.eda.figure_dpi,
                    config.eda.equatorial_latitude_band_degrees
                ))
            elif workflow == "yearly_trends":
                outputs = create_yearly_trend_figures(
                    panel=panel,
                    output_dir=config.eda_figure_dir,
                    start_year=config.eda.start_year,
                    end_year=config.eda.end_year,
                    equatorial_band_degrees=(
                        config.eda.equatorial_latitude_band_degrees
                    ),
                    bootstrap_replicates=config.eda.bootstrap_replicates,
                    confidence_level=(
                        config.eda.bootstrap_confidence_level
                    ),
                    random_seed=config.eda.bootstrap_random_seed,
                    dpi=config.eda.figure_dpi
                )
            elif workflow == "city_slopes":
                outputs = create_city_slope_figures(
                    panel=panel,
                    output_dir=config.eda_figure_dir,
                    start_year=config.eda.start_year,
                    end_year=config.eda.end_year,
                    equatorial_band_degrees=(
                        config.eda.equatorial_latitude_band_degrees
                    ),
                    minimum_years=config.eda.min_city_slope_years,
                    climate_groups=tuple(config.cz_macro_groups),
                    dpi=config.eda.figure_dpi
                )
            elif workflow == "seasonal_contrasts":
                seasonal_results = analyze_seasonal_contrasts(
                    panel=panel,
                    start_year=config.eda.start_year,
                    end_year=config.eda.end_year,
                    equatorial_band_degrees=(
                        config.eda.equatorial_latitude_band_degrees
                    ),
                    northern_summer_months=(
                        config.seasonal_contrast.northern_summer_months
                    ),
                    northern_winter_months=(
                        config.seasonal_contrast.northern_winter_months
                    ),
                    minimum_valid_months_per_season=(
                        config.seasonal_contrast
                        .minimum_valid_months_per_season
                    ),
                    minimum_trend_years=config.eda.min_city_slope_years,
                    climate_groups=tuple(config.cz_macro_groups),
                    bootstrap_replicates=config.eda.bootstrap_replicates,
                    confidence_level=(
                        config.eda.bootstrap_confidence_level
                    ),
                    random_seed=config.eda.bootstrap_random_seed
                )
                outputs = create_seasonal_contrast_outputs(
                    results=seasonal_results,
                    output_dir=config.eda_figure_dir,
                    start_year=config.eda.start_year,
                    end_year=config.eda.end_year,
                    climate_groups=tuple(config.cz_macro_groups),
                    confidence_level=(
                        config.eda.bootstrap_confidence_level
                    ),
                    parquet_compression=config.parquet_compression,
                    dpi=config.eda.figure_dpi
                )
            else:
                outputs = create_climate_month_trend_heatmaps(
                    panel=panel,
                    output_dir=config.eda_figure_dir,
                    start_year=config.eda.start_year,
                    end_year=config.eda.end_year,
                    equatorial_band_degrees=(
                        config.eda.equatorial_latitude_band_degrees
                    ),
                    minimum_years=config.eda.min_city_slope_years,
                    climate_groups=tuple(config.cz_macro_groups),
                    bootstrap_replicates=config.eda.bootstrap_replicates,
                    confidence_level=(
                        config.eda.bootstrap_confidence_level
                    ),
                    random_seed=config.eda.bootstrap_random_seed,
                    dpi=config.eda.figure_dpi
                )
            LOGGER.info(
                "Workflow '%s' created %s output artifacts.",
                workflow,
                len(outputs)
            )
        LOGGER.info("EDA workflow '%s' completed.", workflow_name)
    except Exception:
        LOGGER.exception("EDA workflow '%s' failed.", workflow_name)
        raise

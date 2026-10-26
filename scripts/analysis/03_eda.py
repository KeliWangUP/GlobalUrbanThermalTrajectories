import pandas as pd

from src.config import load_analysis_config
from src.diurnal_eda import (
    aggregate_sensor_city_months,
    create_local_season_figures,
    create_local_season_year_panel_figures,
    create_diurnal_sensor_figures,
    create_year_panel_figures,
    log_sensor_summary
)
from src.eda_logging import configure_eda_logging


def main() -> None:
    config = load_analysis_config()
    logger, _ = configure_eda_logging(config.eda_log_dir)
    logger.info(
        "Starting EDA for %s-%s using %s aggregation.",
        config.eda.start_year,
        config.eda.end_year,
        config.eda.city_month_aggregation
    )

    try:
        if not config.monthly_suhi_file.is_file():
            raise FileNotFoundError(
                "Monthly SUHI master table not found: "
                f"{config.monthly_suhi_file}. "
                "Run 02_build_monthly_suhi.py first."
            )

        logger.info("Input: %s", config.monthly_suhi_file)
        panel = pd.read_parquet(config.monthly_suhi_file)
        if "centroid_lat" not in panel.columns:
            raise ValueError(
                "Monthly SUHI panel is missing centroid_lat. Run "
                "04_merge_city_metadata.py or rebuild it with "
                "02_build_monthly_suhi.py."
            )
        city_months = {}
        for sensor in ("terra", "aqua"):
            city_months[sensor], summary = aggregate_sensor_city_months(
                panel=panel,
                sensor=sensor,
                start_year=config.eda.start_year,
                end_year=config.eda.end_year,
                aggregation=config.eda.city_month_aggregation
            )
            log_sensor_summary(
                sensor,
                summary,
                config.eda.start_year,
                config.eda.end_year
            )

        output_paths = create_diurnal_sensor_figures(
            city_months=city_months,
            output_dir=config.eda_figure_dir,
            start_year=config.eda.start_year,
            end_year=config.eda.end_year,
            aggregation=config.eda.city_month_aggregation,
            dpi=config.eda.figure_dpi
        )
        for output_path in output_paths:
            logger.info("Figure: %s", output_path)

        year_panel_paths = create_year_panel_figures(
            panel=panel,
            output_dir=config.eda_figure_dir,
            start_year=config.eda.start_year,
            end_year=config.eda.end_year,
            dpi=config.eda.figure_dpi
        )
        for output_path in year_panel_paths:
            logger.info("Figure: %s", output_path)

        local_month_paths = create_local_season_figures(
            panel=panel,
            output_dir=config.eda_figure_dir,
            start_year=config.eda.start_year,
            end_year=config.eda.end_year,
            aggregation=config.eda.city_month_aggregation,
            dpi=config.eda.figure_dpi
        )
        for output_path in local_month_paths:
            logger.info("Figure: %s", output_path)

        local_year_paths = create_local_season_year_panel_figures(
            panel=panel,
            output_dir=config.eda_figure_dir,
            start_year=config.eda.start_year,
            end_year=config.eda.end_year,
            dpi=config.eda.figure_dpi
        )
        for output_path in local_year_paths:
            logger.info("Figure: %s", output_path)
        logger.info("EDA completed successfully.")
    except Exception:
        logger.exception("EDA failed.")
        raise


if __name__ == "__main__":
    main()
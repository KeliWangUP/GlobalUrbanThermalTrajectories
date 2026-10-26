import pandas as pd

from src.config import load_analysis_config
from src.diurnal_eda import (
    aggregate_sensor_city_months,
    create_diurnal_sensor_figures,
    print_sensor_summary
)


def main() -> None:
    config = load_analysis_config()
    if not config.monthly_suhi_file.is_file():
        raise FileNotFoundError(
            f"Monthly SUHI master table not found: {config.monthly_suhi_file}. "
            "Run 02_build_monthly_suhi.py first."
        )

    print(f"[INPUT] {config.monthly_suhi_file}")
    panel = pd.read_parquet(config.monthly_suhi_file)
    city_months = {}
    for sensor in ("terra", "aqua"):
        city_months[sensor], summary = aggregate_sensor_city_months(
            panel=panel,
            sensor=sensor,
            start_year=config.eda.start_year,
            end_year=config.eda.end_year,
            aggregation=config.eda.city_month_aggregation
        )
        print_sensor_summary(
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
        print(f"[FIGURE] {output_path}")


if __name__ == "__main__":
    main()
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.diurnal_eda import (
    aggregate_sensor_city_months,
    log_sensor_summary
)
from src.eda_visualizations.common import (
    create_month_grid,
    symmetric_axis_limits
)


LOGGER = logging.getLogger("diurnal_eda")


def create_calendar_month_figures(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    aggregation: str,
    dpi: int
) -> list[Path]:
    """Plot calendar-month day/night SUHI, aggregated across years by city."""
    city_months = {}
    summaries = {}
    for sensor in ("terra", "aqua"):
        city_months[sensor], summaries[sensor] = aggregate_sensor_city_months(
            panel,
            sensor,
            start_year,
            end_year,
            aggregation
        )
        log_sensor_summary(
            sensor,
            summaries[sensor],
            start_year,
            end_year
        )

    lower, upper = symmetric_axis_limits(list(city_months.values()))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for sensor, frame in city_months.items():
        figure = create_month_grid(
            frame=frame,
            sensor=sensor,
            lower=lower,
            upper=upper,
            title=(
                f"{sensor.title()} daytime vs nighttime SUHI "
                f"(calendar month, {aggregation}, {start_year}-{end_year})"
            ),
            x_label="Daytime SUHI (°C)"
        )
        output = output_dir / (
            f"{sensor}_calendar_month_day_vs_night_suhi_{aggregation}_"
            f"{start_year}-{end_year}.png"
        )
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        LOGGER.info("Figure: %s", output)
        outputs.append(output)
    return outputs

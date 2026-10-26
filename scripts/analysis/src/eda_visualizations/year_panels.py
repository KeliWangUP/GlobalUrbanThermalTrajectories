import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.diurnal_eda import (
    MONTH_NAMES,
    log_year_month_summary,
    prepare_sensor_year_month_rows
)
from src.eda_visualizations.common import (
    create_year_grid,
    symmetric_axis_limits
)


LOGGER = logging.getLogger("diurnal_eda")


def create_calendar_year_figures(
    panel: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    dpi: int
) -> list[Path]:
    """Plot one calendar-month figure with a panel for each source year."""
    sensor_data = {}
    for sensor in ("terra", "aqua"):
        sensor_data[sensor] = prepare_sensor_year_month_rows(
            panel,
            sensor,
            start_year,
            end_year
        )
    lower, upper = symmetric_axis_limits(
        [data[0] for data in sensor_data.values()]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for sensor, (rows, summary) in sensor_data.items():
        for month in range(1, 13):
            log_year_month_summary(
                sensor,
                month,
                summary,
                start_year,
                end_year
            )
            figure = create_year_grid(
                rows=rows,
                summary=summary,
                sensor=sensor,
                month=month,
                start_year=start_year,
                end_year=end_year,
                lower=lower,
                upper=upper,
                title=(
                    f"{sensor.title()} {MONTH_NAMES[month - 1]}: "
                    "daytime vs nighttime SUHI by year"
                ),
                x_label="Daytime SUHI (°C)"
            )
            sensor_dir = output_dir / "by_year" / sensor
            sensor_dir.mkdir(parents=True, exist_ok=True)
            output = sensor_dir / (
                f"{sensor}_{MONTH_NAMES[month - 1].lower()}_"
                f"day_vs_night_suhi_by_year_{start_year}-{end_year}.png"
            )
            figure.savefig(output, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            LOGGER.info("Figure: %s", output)
            outputs.append(output)
    return outputs

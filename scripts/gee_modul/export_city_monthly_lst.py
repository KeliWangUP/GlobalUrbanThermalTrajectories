from typing import Any

import ee

from src.export_utils import run_annual_exports


def export_city(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> None:
    """Export annual LST statistics for configured city boundaries."""
    city_config = config["cities"]
    regions = ee.FeatureCollection(city_config["asset"])
    run_annual_exports(
        dataset_name=config["export"]["city_dataset_name"],
        regions=regions,
        config=config,
        projection=projection,
        scale=scale
    )

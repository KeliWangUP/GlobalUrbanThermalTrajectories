from typing import Any

import ee

from src.export_utils import run_annual_exports
from src.rural_reference_utils import (
    aggregate_rural_mask_to_modis,
    build_all_rural_halos,
    build_global_stable_rural_binary,
    build_global_urban_exclusion_mask
)


def build_rural_inputs(
    config: dict[str, Any],
    projection: ee.Projection,
    city_id: int | None = None
) -> tuple[ee.FeatureCollection, ee.Image]:
    """Build rural halo regions and the MODIS-grid rural suitability mask."""
    city_config = config["cities"]
    rural_config = config["rural"]
    cities = ee.FeatureCollection(city_config["asset"])

    halo_cities = cities
    if city_id is not None:
        halo_cities = cities.filter(
            ee.Filter.eq(city_config["id_field"], city_id)
        )

    rural_halos = build_all_rural_halos(
        cities=halo_cities,
        inner_buffer_m=rural_config["inner_buffer_m"],
        outer_buffer_m=rural_config["outer_buffer_m"],
        city_id_field=city_config["id_field"],
        max_error_m=rural_config["max_error_m"]
    )

    urban_mask_source = ee.Image(rural_config["gaia_asset"])

    gaia_projection = (
        urban_mask_source
        .select("change_year_index")
        .projection()
    )

    urban_exclusion = build_global_urban_exclusion_mask(
        cities=cities,
        buffer_m=rural_config["inner_buffer_m"],
        max_error_m=rural_config["max_error_m"],
        reference_projection=gaia_projection
    )
    
    rural_binary = build_global_stable_rural_binary(
        urban_mask_source=urban_mask_source,
        dem_mask_source=ee.ImageCollection(rural_config["dem_collection"]),
        global_exclusion_mask=urban_exclusion,
        max_slope_deg=rural_config["max_slope_deg"],
    )
    _, rural_modis_mask = aggregate_rural_mask_to_modis(
        rural_binary=rural_binary,
        modis_projection=projection,
        minimum_rural_fraction=(
            rural_config["minimum_rural_fraction_per_modis_pixel"]
        )
    )
    return rural_halos, rural_modis_mask


def export_rural(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> None:
    """Export annual LST statistics for rural halo regions."""
    rural_halos, rural_modis_mask = build_rural_inputs(config, projection)
    run_annual_exports(
        dataset_name=config["export"]["rural_dataset_name"],
        regions=rural_halos,
        config=config,
        projection=projection,
        scale=scale,
        spatial_mask=rural_modis_mask
    )

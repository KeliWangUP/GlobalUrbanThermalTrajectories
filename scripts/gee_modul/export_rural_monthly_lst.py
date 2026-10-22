from typing import Any

import ee

from src.batch_utils import (
    load_city_batch,
    resolve_batch_indices,
    verify_city_batches
)
from src.export_utils import run_annual_exports
from src.rural_reference_utils import (
    build_all_rural_halos,
    build_rural_modis_mask
)


def build_rural_halos(
    config: dict[str, Any],
    cities: ee.FeatureCollection
) -> ee.FeatureCollection:
    """Build rural halos for the given cities."""
    rural_config = config["rural"]
    return build_all_rural_halos(
        cities=cities,
        inner_buffer_m=rural_config["inner_buffer_m"],
        outer_buffer_m=rural_config["outer_buffer_m"],
        city_id_field=config["cities"]["id_field"],
        max_error_m=rural_config["max_error_m"]
    )


def build_rural_mask(
    config: dict[str, Any],
    projection: ee.Projection
) -> ee.Image:
    """Build the MODIS-grid rural mask; exclusion uses all cities."""
    rural_config = config["rural"]
    return build_rural_modis_mask(
        cities=ee.FeatureCollection(config["cities"]["asset"]),
        urban_mask_source=ee.Image(rural_config["gaia_asset"]),
        dem_mask_source=ee.ImageCollection(rural_config["dem_collection"]),
        modis_projection=projection,
        inner_buffer_m=rural_config["inner_buffer_m"],
        max_error_m=rural_config["max_error_m"],
        max_slope_deg=rural_config["max_slope_deg"],
        minimum_rural_fraction=(
            rural_config["minimum_rural_fraction_per_modis_pixel"]
        ),
        max_input_pixels=rural_config["max_input_pixels_per_modis_pixel"]
    )


def export_rural(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> None:
    """Submit one annual CSV export per selected city batch and year."""
    batch_indices = resolve_batch_indices(config)
    verify_city_batches(config, batch_indices)

    rural_mask = build_rural_mask(config, projection)
    dataset_name = config["export"]["rural_dataset_name"]

    submitted = 0
    for batch_index in batch_indices:
        print(f"\n[INFO] Rural batch {batch_index:02d}")
        tasks = run_annual_exports(
            dataset_name=f"{dataset_name}_batch{batch_index:02d}",
            regions=build_rural_halos(
                config,
                load_city_batch(config, batch_index)
            ),
            config=config,
            projection=projection,
            scale=scale,
            spatial_mask=rural_mask
        )
        submitted += len(tasks)

    print(
        f"\n[INFO] Submitted {submitted} rural export tasks for "
        f"{len(batch_indices)} batches."
    )

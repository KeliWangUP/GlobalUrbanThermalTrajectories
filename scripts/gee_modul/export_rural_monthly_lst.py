from typing import Any

import ee

from src.batch_utils import (
    TILE_ID_PROPERTY,
    filter_nonempty_geometries,
    get_occupied_subtile_ids,
    get_tile_city_counts,
    resolve_batch_indices
)
from src.export_utils import (
    asset_exists,
    list_image_asset_names,
    run_annual_exports
)
from src.rural_reference_utils import RURAL_FRACTION_BAND


def load_rural_halos(
    config: dict[str, Any],
    tile_id: str
) -> ee.FeatureCollection:
    """Load the precomputed rural halos assigned to one MODIS tile."""
    return (
        filter_nonempty_geometries(
            ee.FeatureCollection(
                config["rural"]["static_assets"]["halo_asset"]
            ).filter(
                ee.Filter.eq(TILE_ID_PROPERTY, tile_id)
            ),
            config["rural"]["max_error_m"]
        )
    )


def build_rural_mask(
    config: dict[str, Any],
    scale: float,
    regions: ee.FeatureCollection
) -> ee.Image:
    """Load cached fractions intersecting the selected halo regions."""
    rural_config = config["rural"]
    static_config = rural_config["static_assets"]
    max_error_m = rural_config["max_error_m"]
    processing_region = (
        regions
        .geometry(max_error_m)
        .bounds(max_error_m)
        .buffer(
            static_config["processing_buffer_modis_pixels"] * scale,
            max_error_m
        )
    )
    fraction = (
        ee.ImageCollection(static_config["fraction_collection_asset"])
        .filterBounds(processing_region)
        .select(RURAL_FRACTION_BAND)
        .mosaic()
    )
    fraction_scale = static_config["fraction_value_scale"]
    minimum_fraction = rural_config[
        "minimum_rural_fraction_per_modis_pixel"
    ]
    return (
        fraction
        .gte(round(minimum_fraction * fraction_scale))
        .rename("rural_modis_mask")
    )


def verify_static_rural_assets(
    config: dict[str, Any],
    projection: ee.Projection,
    halos: ee.FeatureCollection
) -> None:
    """Require all fraction subtiles intersecting the stored halos."""
    static_config = config["rural"]["static_assets"]
    halo_asset = static_config["halo_asset"]
    fraction_collection = static_config["fraction_collection_asset"]
    if not asset_exists(halo_asset):
        raise FileNotFoundError(
            f"Rural halo asset not found: {halo_asset}. Run "
            "build_rural_halos_asset.py first."
        )
    if not asset_exists(fraction_collection):
        raise FileNotFoundError(
            f"Rural fraction collection not found: {fraction_collection}. "
            "Run build_rural_fraction_assets.py first."
        )

    expected = set(get_occupied_subtile_ids(config, projection, halos))
    existing = list_image_asset_names(fraction_collection)
    missing = sorted(expected - existing)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} rural fraction subtile assets are missing "
            f"(for example: {preview}). Run build_rural_fraction_assets.py "
            "until all subtile assets exist."
        )


def export_rural(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> None:
    """Submit annual CSV exports for selected precomputed tile halos."""
    static_config = config["rural"]["static_assets"]
    halo_asset = static_config["halo_asset"]
    if not asset_exists(halo_asset):
        raise FileNotFoundError(
            f"Rural halo asset not found: {halo_asset}. Run "
            "build_rural_halos_asset.py first."
        )

    halos = filter_nonempty_geometries(
        ee.FeatureCollection(halo_asset),
        config["rural"]["max_error_m"]
    )
    tile_ids = list(get_tile_city_counts(halos))
    batch_indices = resolve_batch_indices(config, tile_ids)
    selected_tile_ids = [tile_ids[index] for index in batch_indices]
    selected_halos = halos.filter(
        ee.Filter.inList(TILE_ID_PROPERTY, selected_tile_ids)
    )
    verify_static_rural_assets(config, projection, selected_halos)

    submitted = 0
    dataset_name = config["export"]["rural_dataset_name"]
    for batch_index in batch_indices:
        tile_id = tile_ids[batch_index]
        print(
            f"\n[INFO] Rural batch {batch_index:03d} "
            f"(MODIS tile {tile_id})"
        )
        regions = load_rural_halos(config, tile_id)
        rural_mask = build_rural_mask(
            config,
            scale,
            regions
        )
        tasks = run_annual_exports(
            dataset_name=f"{dataset_name}_{tile_id}",
            regions=regions,
            config=config,
            projection=projection,
            scale=scale,
            spatial_mask=rural_mask
        )
        submitted += len(tasks)

    print(
        f"\n[INFO] Submitted {submitted} rural export tasks for "
        f"{len(batch_indices)} MODIS tile batches."
    )

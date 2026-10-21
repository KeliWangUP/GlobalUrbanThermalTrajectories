import ee

from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.rural_reference_utils import (
    build_all_rural_halos,
    build_rural_fraction_image,
    get_modis_chunk_export_grid,
    get_rural_modis_chunk_ids
)


def ensure_fraction_collection(collection_id: str) -> None:
    """Create the destination ImageCollection if it does not exist."""

    try:
        asset = ee.data.getAsset(collection_id)
    except ee.EEException:
        asset = ee.data.createAsset(
            {"type": "IMAGE_COLLECTION"},
            collection_id
        )

    if asset.get("type") != "IMAGE_COLLECTION":
        raise ValueError(
            "Rural fraction destination must be an ImageCollection asset: "
            f"{collection_id}"
        )


def require_tested_chunk(config: dict) -> None:
    """Require a successful single-chunk export with current settings."""

    rural_config = config["rural"]
    test_asset_id = rural_config["static_assets"]["test_fraction_asset"]
    try:
        asset = ee.data.getAsset(test_asset_id)
    except ee.EEException as exc:
        raise RuntimeError(
            "Global fraction export is gated on a successful test chunk. "
            "Run test_rural_fraction_chunk.py and wait for its task to "
            f"complete first. Test asset: {test_asset_id}"
        ) from exc

    if asset.get("type") != "IMAGE":
        raise ValueError(
            f"Rural fraction test asset is not an IMAGE: {test_asset_id}"
        )

    test_config = config["test"]
    grid_config = config["reference_grid"]["tile_grid"]
    expected = {
        "test_city_id": test_config["city_id"],
        "test_chunk_size_pixels": grid_config["chunk_size_pixels"],
        "test_max_input_pixels_per_modis_pixel": (
            rural_config["max_input_pixels_per_modis_pixel"]
        ),
        "test_minimum_rural_fraction_per_modis_pixel": (
            rural_config["minimum_rural_fraction_per_modis_pixel"]
        ),
        "test_inner_buffer_m": rural_config["inner_buffer_m"],
        "test_outer_buffer_m": rural_config["outer_buffer_m"],
        "test_max_slope_deg": rural_config["max_slope_deg"],
        "test_max_error_m": rural_config["max_error_m"],
        "test_gaia_asset": rural_config["gaia_asset"],
        "test_dem_collection": rural_config["dem_collection"]
    }
    actual = asset.get("properties", {})
    mismatches = {
        key: (expected_value, actual.get(key))
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "Rural fraction test asset does not match the current "
            f"configuration: {mismatches}. Rerun test_rural_fraction_chunk.py "
            "with a new test_fraction_asset path."
        )


def main() -> None:
    config = load_config()
    initialize_gee(config)
    require_tested_chunk(config)

    projection, scale = get_modis_reference_grid(config)
    city_config = config["cities"]
    rural_config = config["rural"]
    grid_config = config["reference_grid"]["tile_grid"]
    chunk_size = grid_config["chunk_size_pixels"]
    max_input_pixels = rural_config[
        "max_input_pixels_per_modis_pixel"
    ]
    estimated_chunk_inputs = (
        (chunk_size + 2) ** 2
        * max_input_pixels
    )
    if estimated_chunk_inputs >= 2 ** 31:
        raise ValueError(
            "Configured MODIS chunk may exceed Earth Engine's 2^31 input "
            "pixel limit. Reduce chunk_size_pixels or "
            "max_input_pixels_per_modis_pixel."
        )
    collection_id = rural_config["static_assets"][
        "fraction_collection_asset"
    ]
    cities = ee.FeatureCollection(city_config["asset"])
    rural_halos = build_all_rural_halos(
        cities=cities,
        inner_buffer_m=rural_config["inner_buffer_m"],
        outer_buffer_m=rural_config["outer_buffer_m"],
        city_id_field=city_config["id_field"],
        max_error_m=rural_config["max_error_m"]
    )

    projection_info = projection.getInfo()
    chunk_ids = get_rural_modis_chunk_ids(
        rural_halos=rural_halos,
        max_error_m=rural_config["max_error_m"],
        projection_crs=projection_info["crs"],
        scale=scale,
        tile_grid=grid_config
    )
    ensure_fraction_collection(collection_id)
    existing_chunk_ids = set(
        ee.ImageCollection(collection_id)
        .aggregate_array("chunk_id")
        .getInfo()
        or []
    )
    pending_chunk_ids = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id not in existing_chunk_ids
    ]
    if not pending_chunk_ids:
        print(
            f"[INFO] All {len(chunk_ids)} rural MODIS fraction chunks "
            "already exist."
        )
        return

    rural_fraction = build_rural_fraction_image(
        cities=cities,
        urban_mask_source=ee.Image(rural_config["gaia_asset"]),
        dem_mask_source=ee.ImageCollection(
            rural_config["dem_collection"]
        ),
        modis_projection=projection,
        inner_buffer_m=rural_config["inner_buffer_m"],
        max_error_m=rural_config["max_error_m"],
        max_slope_deg=rural_config["max_slope_deg"],
        minimum_rural_fraction=(
            rural_config["minimum_rural_fraction_per_modis_pixel"]
        ),
        max_input_pixels=max_input_pixels
    )

    print(
        f"[INFO] Rural MODIS chunks intersecting coverage: "
        f"{len(chunk_ids)}; new chunk exports: {len(pending_chunk_ids)}."
    )
    for chunk_id in pending_chunk_ids:
        chunk_region, transform = get_modis_chunk_export_grid(
            chunk_id=chunk_id,
            projection_crs=projection_info["crs"],
            scale=scale,
            tile_grid=grid_config
        )
        chunk_asset_id = f"{collection_id}/{chunk_id}"
        chunk_image = rural_fraction.clip(chunk_region).set(
            "chunk_id",
            chunk_id
        )
        task = ee.batch.Export.image.toAsset(
            image=chunk_image,
            description=f"rural_fraction_{chunk_id}",
            assetId=chunk_asset_id,
            region=chunk_region,
            crs=projection_info["crs"],
            crsTransform=transform,
            maxPixels=(chunk_size + 2) ** 2,
            pyramidingPolicy={".default": "mean"}
        )
        task.start()
        print(f"[EXPORT] {chunk_asset_id}")
        print(f"[TASK] {task.id}")

    print(
        "[NEXT] Wait for all chunk tasks to complete, then run "
        "build_rural_halos_asset.py."
    )


if __name__ == "__main__":
    main()

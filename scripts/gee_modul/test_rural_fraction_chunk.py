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


def main() -> None:
    config = load_config()
    initialize_gee(config)

    projection, scale = get_modis_reference_grid(config)
    projection_info = projection.getInfo()
    city_config = config["cities"]
    rural_config = config["rural"]
    test_config = config["test"]
    static_assets = rural_config["static_assets"]
    grid_config = config["reference_grid"]["tile_grid"]
    city_id = test_config["city_id"]
    asset_id = static_assets["test_fraction_asset"]

    try:
        ee.data.getAsset(asset_id)
    except ee.EEException:
        pass
    else:
        raise FileExistsError(
            "The rural fraction test asset already exists. Use a new "
            "test_fraction_asset path or remove only that test asset before "
            "rerunning this test."
        )

    cities = ee.FeatureCollection(city_config["asset"])
    selected_city = cities.filter(
        ee.Filter.eq(city_config["id_field"], city_id)
    )
    city_count = selected_city.size().getInfo()
    if city_count != 1:
        raise ValueError(
            f"Expected exactly one city with "
            f"{city_config['id_field']}={city_id}, found {city_count}."
        )

    rural_halos = build_all_rural_halos(
        cities=selected_city,
        inner_buffer_m=rural_config["inner_buffer_m"],
        outer_buffer_m=rural_config["outer_buffer_m"],
        city_id_field=city_config["id_field"],
        max_error_m=rural_config["max_error_m"]
    )
    chunk_ids = get_rural_modis_chunk_ids(
        rural_halos=rural_halos,
        max_error_m=rural_config["max_error_m"],
        projection_crs=projection_info["crs"],
        scale=scale,
        tile_grid=grid_config
    )
    chunk_id = chunk_ids[0]
    chunk_region, transform = get_modis_chunk_export_grid(
        chunk_id=chunk_id,
        projection_crs=projection_info["crs"],
        scale=scale,
        tile_grid=grid_config
    )

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
        max_input_pixels=(
            rural_config["max_input_pixels_per_modis_pixel"]
        )
    )

    test_image = rural_fraction.clip(chunk_region).set({
        "chunk_id": chunk_id,
        "test_city_id": city_id,
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
    })
    chunk_size = grid_config["chunk_size_pixels"]
    task = ee.batch.Export.image.toAsset(
        image=test_image,
        description=f"rural_fraction_test_{city_id}_{chunk_id}",
        assetId=asset_id,
        region=chunk_region,
        crs=projection_info["crs"],
        crsTransform=transform,
        maxPixels=(chunk_size + 2) ** 2,
        pyramidingPolicy={".default": "mean"}
    )
    task.start()
    print(f"[TEST] City ID: {city_id}")
    print(f"[TEST] Chunk: {chunk_id}; total candidate chunks: {len(chunk_ids)}")
    print(f"[TEST] Asset: {asset_id}")
    print(f"[TASK] {task.id}")
    print(
        "[NEXT] Confirm this task reaches COMPLETED before running "
        "build_rural_fraction_asset.py."
    )


if __name__ == "__main__":
    main()

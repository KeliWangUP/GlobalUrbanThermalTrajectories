import ee

from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.rural_reference_utils import build_rural_fraction_image


def main() -> None:
    config = load_config()
    initialize_gee(config)

    projection, _ = get_modis_reference_grid(config)
    city_config = config["cities"]
    rural_config = config["rural"]
    cities = ee.FeatureCollection(city_config["asset"])

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

    asset_id = rural_config["static_assets"]["fraction_asset"]
    projection_info = projection.getInfo()
    region = ee.Geometry.Rectangle(
        [-180, -90, 180, 90],
        proj="EPSG:4326",
        geodesic=False
    )
    task = ee.batch.Export.image.toAsset(
        image=rural_fraction,
        description=asset_id.rsplit("/", 1)[-1],
        assetId=asset_id,
        region=region,
        crs=projection_info["crs"],
        crsTransform=projection_info["transform"],
        maxPixels=10_000_000_000_000,
        pyramidingPolicy={".default": "mean"}
    )
    task.start()
    print(f"[EXPORT] Rural fraction asset: {asset_id}")
    print(f"[TASK] {task.id}")
    print(
        "[NEXT] Wait for this task to complete, then run "
        "build_rural_halos_asset.py."
    )


if __name__ == "__main__":
    main()

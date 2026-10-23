import math
from typing import Any

import ee

from src.batch_utils import filter_nonempty_geometries
from src.export_utils import get_modis_reference_grid, initialize_gee, load_config


def projection_summary(
    label: str,
    image: ee.Image,
    region: ee.Geometry,
    max_error_m: int
) -> None:
    print(f"\n[PROJECTION] {label}", flush=True)
    projection = image.projection()
    try:
        info = projection.getInfo()
        nominal_scale = projection.nominalScale().getInfo()
        bounds = (
            region
            .bounds(maxError=max_error_m, proj=projection)
            .coordinates()
            .getInfo()[0]
        )
    except ee.EEException as exc:
        print(f"[FAILED] Projection inspection for {label}: {exc}", flush=True)
        raise

    transform = info["transform"]
    x_values = [point[0] for point in bounds]
    y_values = [point[1] for point in bounds]
    width = max(x_values) - min(x_values)
    height = max(y_values) - min(y_values)
    columns = math.ceil(width / abs(transform[0]))
    rows = math.ceil(height / abs(transform[4]))
    print(f"  CRS: {info['crs']}", flush=True)
    print(f"  transform: {transform}", flush=True)
    print(f"  nominal scale: {nominal_scale}", flush=True)
    print(
        f"  region bounds estimate: {columns} x {rows} pixels",
        flush=True
    )


def evaluate_stage(
    label: str,
    image: ee.Image,
    region: ee.Geometry,
    projection_info: dict[str, Any],
    max_pixels: int,
    tile_scale: int
) -> None:
    print(f"\n[EVALUATE] {label} over the complete city halo", flush=True)
    try:
        result = (
            image
            .reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.count(),
                    sharedInputs=True
                ),
                geometry=region,
                crs=projection_info["crs"],
                crsTransform=projection_info["transform"],
                maxPixels=max_pixels,
                tileScale=tile_scale
            )
            .getInfo()
        )
    except ee.EEException as exc:
        print(f"[FAILED] Evaluation stopped at {label}: {exc}", flush=True)
        raise
    print(f"  result: {result}", flush=True)


def main() -> None:
    config = load_config()
    initialize_gee(config)

    rural_config = config["rural"]
    static_config = rural_config["static_assets"]
    city_id = config["test"]["fraction_diagnostic_city_id"]
    id_field = config["cities"]["id_field"]
    max_error_m = rural_config["max_error_m"]
    tile_scale = config["processing"]["tile_scale"]
    modis_projection, scale = get_modis_reference_grid(config)
    modis_projection_info = modis_projection.getInfo()

    halo_asset = static_config["halo_asset"]
    halos = filter_nonempty_geometries(
        ee.FeatureCollection(halo_asset),
        max_error_m
    )
    city_halos = halos.filter(ee.Filter.eq(id_field, city_id))
    halo_count = city_halos.size().getInfo()
    if halo_count != 1:
        raise ValueError(
            f"Expected one non-empty halo for city {city_id}, "
            f"found {halo_count}."
        )

    halo_geometry = ee.Feature(city_halos.first()).geometry()
    area_km2 = (
        halo_geometry
        .area(maxError=max_error_m)
        .divide(1_000_000)
        .getInfo()
    )
    modis_bounds = (
        halo_geometry
        .bounds(maxError=max_error_m, proj=modis_projection)
        .coordinates()
        .getInfo()[0]
    )
    x_values = [point[0] for point in modis_bounds]
    y_values = [point[1] for point in modis_bounds]
    modis_transform = modis_projection_info["transform"]
    expected_columns = math.ceil(
        (max(x_values) - min(x_values)) / abs(modis_transform[0])
    )
    expected_rows = math.ceil(
        (max(y_values) - min(y_values)) / abs(modis_transform[4])
    )
    print(
        f"[CITY] ID={city_id}; halo area={area_km2:.2f} km^2; "
        f"expected MODIS bounds={expected_columns} x {expected_rows} pixels",
        flush=True
    )

    gaia = ee.Image(rural_config["gaia_asset"])
    stable_nonurban = (
        gaia
        .select("change_year_index")
        .unmask(0)
        .eq(0)
        .rename("diagnostic")
    )
    dem_collection = ee.ImageCollection(rural_config["dem_collection"])
    first_dem = ee.Image(dem_collection.first()).select("DEM")
    dem_projection = first_dem.projection()
    dem = (
        dem_collection
        .select("DEM")
        .mosaic()
        .setDefaultProjection(dem_projection)
    )
    slope_ok = ee.Terrain.slope(dem).lt(
        rural_config["max_slope_deg"]
    ).rename("diagnostic")
    non_water = (
        dem_collection
        .select("WBM")
        .mosaic()
        .setDefaultProjection(dem_projection)
        .eq(0)
        .rename("diagnostic")
    )
    eligibility = (
        stable_nonurban
        .And(slope_ok)
        .And(non_water)
        .unmask(0)
        .rename("diagnostic")
    )
    processing_region = halo_geometry.buffer(
        static_config["processing_buffer_modis_pixels"] * scale,
        max_error_m
    )
    clipped_eligibility = eligibility.clip(processing_region)
    reduced_fraction = clipped_eligibility.toFloat().reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=static_config["max_source_pixels_per_modis_pixel"]
    )
    modis_fraction = reduced_fraction.reproject(
        crs=modis_projection
    )
    scaled_fraction = (
        modis_fraction
        .multiply(static_config["fraction_value_scale"])
        .floor()
        .toUint16()
        .rename("diagnostic")
        .clip(halo_geometry)
    )

    stages = [
        ("GAIA stable-nonurban mask", stable_nonurban),
        ("GLO30 DEM mosaic", dem.rename("diagnostic")),
        ("GLO30 slope mask", slope_ok),
        ("GLO30 water mask", non_water),
        ("combined eligibility", eligibility),
        ("eligibility clipped to buffered halo", clipped_eligibility),
        ("reduceResolution output", reduced_fraction.rename("diagnostic")),
        ("MODIS reproject output", modis_fraction.rename("diagnostic")),
        ("scaled and halo-clipped fraction", scaled_fraction)
    ]

    for label, image in stages:
        projection_summary(
            label=label,
            image=image,
            region=halo_geometry,
            max_error_m=max_error_m
        )
        evaluate_stage(
            label=label,
            image=image,
            region=halo_geometry,
            projection_info=modis_projection_info,
            max_pixels=static_config["export_max_pixels"],
            tile_scale=tile_scale
        )

    print(
        "\n[DONE] Diagnostic evaluations completed. "
        "No export or Earth Engine task was created.",
        flush=True
    )


if __name__ == "__main__":
    main()

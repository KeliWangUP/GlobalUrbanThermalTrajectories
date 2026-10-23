from typing import Any

import ee

from src.batch_utils import filter_nonempty_geometries
from src.export_utils import get_modis_reference_grid, initialize_gee, load_config


def projection_summary(
    label: str,
    image: ee.Image
) -> None:
    print(f"\n[PROJECTION] {label}", flush=True)
    projection = image.projection()
    try:
        info = projection.getInfo()
        nominal_scale = projection.nominalScale().getInfo()
    except ee.EEException as exc:
        print(f"[FAILED] Projection inspection for {label}: {exc}", flush=True)
        raise

    transform = info["transform"]
    print(f"  CRS: {info['crs']}", flush=True)
    print(f"  transform: {transform}", flush=True)
    print(f"  nominal scale: {nominal_scale}", flush=True)


def print_modis_pixel_bounds(
    region: ee.Geometry,
    projection: ee.Projection,
    projection_info: dict[str, Any],
    max_pixels: int,
    tile_scale: int
) -> None:
    print("\n[GRID] Counting the halo's MODIS-grid pixel bounds", flush=True)
    pixel_coordinates = ee.Image.pixelCoordinates(projection)
    try:
        bounds = (
            pixel_coordinates
            .reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=region,
                crs=projection_info["crs"],
                crsTransform=projection_info["transform"],
                maxPixels=max_pixels,
                tileScale=tile_scale
            )
            .getInfo()
        )
    except ee.EEException as exc:
        print(f"[FAILED] MODIS pixel-bound calculation: {exc}", flush=True)
        raise

    required_keys = ("x_min", "x_max", "y_min", "y_max")
    if any(bounds.get(key) is None for key in required_keys):
        raise ValueError(
            f"MODIS pixel bounds were incomplete: {bounds}"
        )
    columns = int(bounds["x_max"] - bounds["x_min"] + 1)
    rows = int(bounds["y_max"] - bounds["y_min"] + 1)
    print(
        f"  x={bounds['x_min']}..{bounds['x_max']}; "
        f"y={bounds['y_min']}..{bounds['y_max']}",
        flush=True
    )
    print(f"  MODIS pixel bounds: {columns} x {rows}", flush=True)


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
    print(
        f"[CITY] ID={city_id}; halo area={area_km2:.2f} km^2",
        flush=True
    )
    print_modis_pixel_bounds(
        region=halo_geometry,
        projection=modis_projection,
        projection_info=modis_projection_info,
        max_pixels=static_config["export_max_pixels"],
        tile_scale=tile_scale
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
            image=image
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

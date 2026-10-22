from typing import Any

import ee

from src.batch_utils import TILE_ID_PROPERTY

RURAL_FRACTION_BAND = "rural_fraction_scaled"


def build_rural_halo(
    feature: ee.Feature,
    all_cities: ee.FeatureCollection,
    inner_buffer_m: int,
    outer_buffer_m: int,
    city_id_field: str,
    max_error_m: int
) -> ee.Feature:
    """Build a city halo excluding every nearby city's 2 km influence area."""
    feature = ee.Feature(feature)
    geometry = feature.geometry()
    inner = geometry.buffer(inner_buffer_m, max_error_m)
    outer = geometry.buffer(outer_buffer_m, max_error_m)
    nearby_cities = all_cities.filterBounds(
        outer.buffer(inner_buffer_m, max_error_m)
    )
    excluded_area = nearby_cities.map(
        lambda city: ee.Feature(city).buffer(
            inner_buffer_m,
            max_error_m
        )
    ).geometry(max_error_m)
    rural_halo = (
        outer
        .difference(inner, max_error_m)
        .difference(excluded_area, max_error_m)
    )

    return (
        ee.Feature(rural_halo)
        .copyProperties(feature, [city_id_field, TILE_ID_PROPERTY])
    )


def build_all_rural_halos(
    cities: ee.FeatureCollection,
    inner_buffer_m: int,
    outer_buffer_m: int,
    city_id_field: str,
    max_error_m: int
) -> ee.FeatureCollection:
    """Build all city halos, subtracting overlapping city buffers."""
    return cities.map(
        lambda feature: build_rural_halo(
            feature=feature,
            all_cities=cities,
            inner_buffer_m=inner_buffer_m,
            outer_buffer_m=outer_buffer_m,
            city_id_field=city_id_field,
            max_error_m=max_error_m
        )
    )


def build_intermediate_modis_projection(
    modis_projection: ee.Projection,
    scale_divisor: int
) -> ee.Projection:
    """Create a MODIS-aligned projection with finer, nested pixels."""
    if type(scale_divisor) is not int or scale_divisor < 1:
        raise ValueError("Intermediate scale divisor must be a positive integer.")

    projection_info = modis_projection.getInfo()
    transform = projection_info["transform"]
    if (
        len(transform) != 6
        or transform[0] <= 0
        or transform[4] >= 0
        or transform[1] != 0
        or transform[3] != 0
    ):
        raise ValueError(
            "MODIS projection must provide an axis-aligned affine transform."
        )

    intermediate_transform = [
        transform[0] / scale_divisor,
        0,
        transform[2],
        0,
        transform[4] / scale_divisor,
        transform[5]
    ]
    return ee.Projection(
        projection_info["crs"],
        intermediate_transform
    )


def build_local_rural_eligibility(
    urban_mask_source: ee.Image,
    dem_mask_source: ee.ImageCollection,
    max_slope_deg: float
) -> ee.Image:
    """Build fine-grid suitability from GAIA and GLO30 only."""
    stable_nonurban = (
        urban_mask_source
        .select("change_year_index")
        .unmask(0)
        .eq(0)
    )

    dem_projection = (
        ee.Image(dem_mask_source.first())
        .select("DEM")
        .projection()
    )
    dem = (
        dem_mask_source
        .select("DEM")
        .mosaic()
        .setDefaultProjection(dem_projection)
    )
    slope_ok = ee.Terrain.slope(dem).lt(max_slope_deg)
    non_water = (
        dem_mask_source
        .select("WBM")
        .mosaic()
        .setDefaultProjection(dem_projection)
        .eq(0)
    )

    return (
        stable_nonurban
        .And(slope_ok)
        .And(non_water)
        .unmask(0)
        .rename("rural_eligibility")
        .byte()
    )


def build_rural_fraction_for_subtile(
    config: dict[str, Any],
    modis_projection: ee.Projection,
    intermediate_projection: ee.Projection,
    subtile_region: ee.Geometry,
    scale: float
) -> ee.Image:
    """Aggregate fine suitability through a local nested MODIS grid."""
    rural_config = config["rural"]
    static_config = rural_config["static_assets"]
    max_error_m = rural_config["max_error_m"]
    processing_region = subtile_region.buffer(
        static_config["processing_buffer_modis_pixels"] * scale,
        max_error_m
    )

    fine_eligibility = (
        build_local_rural_eligibility(
            urban_mask_source=ee.Image(rural_config["gaia_asset"]),
            dem_mask_source=ee.ImageCollection(
                rural_config["dem_collection"]
            ),
            max_slope_deg=rural_config["max_slope_deg"]
        )
        .clip(processing_region)
    )
    intermediate_fraction = (
        fine_eligibility
        .toFloat()
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=static_config[
                "max_source_pixels_per_intermediate_pixel"
            ]
        )
        .reproject(crs=intermediate_projection)
        .clip(processing_region)
    )
    modis_fraction = (
        intermediate_fraction
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=static_config[
                "max_intermediate_pixels_per_modis_pixel"
            ]
        )
        .reproject(crs=modis_projection)
    )

    fraction_scale = static_config["fraction_value_scale"]
    if type(fraction_scale) is not int or fraction_scale < 1:
        raise ValueError(
            "rural.static_assets.fraction_value_scale must be a positive integer."
        )
    return (
        modis_fraction
        .multiply(fraction_scale)
        .floor()
        .toUint16()
        .rename(RURAL_FRACTION_BAND)
        .clip(subtile_region)
    )

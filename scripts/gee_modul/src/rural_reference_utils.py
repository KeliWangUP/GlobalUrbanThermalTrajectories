import re
from typing import Any

import ee

from src.batch_utils import (
    get_occupied_subtile_ids
)

RURAL_FRACTION_BAND = "rural_fraction_scaled"


def apply_rural_fraction_threshold(
    config: dict[str, Any],
    fraction: ee.Image
) -> ee.Image:
    """Threshold a scaled rural fraction image into a MODIS binary mask."""
    rural_config = config["rural"]
    static_config = rural_config["static_assets"]
    fraction_scale = static_config["fraction_value_scale"]
    minimum_fraction = rural_config[
        "minimum_rural_fraction_per_modis_pixel"
    ]
    return (
        fraction
        .gte(round(minimum_fraction * fraction_scale))
        .rename("rural_modis_mask")
    )


def rural_city_fraction_asset_id(
    config: dict[str, Any],
    city_id: int | str
) -> str:
    """Return the fraction image asset path for a city ID."""
    token = str(city_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise ValueError(
            f"City ID cannot be used as an Earth Engine asset name: {city_id}"
        )
    collection = config["rural"]["static_assets"][
        "city_fraction_collection_asset"
    ]
    return f"{collection}/{token}"


def rural_city_fraction_chunk_asset_id(
    config: dict[str, Any],
    city_id: int | str,
    subtile_id: str
) -> str:
    """Return the asset path for one MODIS-aligned city fraction chunk."""
    if not re.fullmatch(r"x\d{3}y\d{3}", subtile_id):
        raise ValueError(f"Invalid MODIS subtile ID: {subtile_id}")
    return f"{rural_city_fraction_asset_id(config, city_id)}_{subtile_id}"


def load_rural_city_fraction(
    config: dict[str, Any],
    city_id: int | str
) -> ee.Image:
    """Mosaic all stored fraction images belonging to a city."""
    collection = config["rural"]["static_assets"][
        "city_fraction_collection_asset"
    ]
    return (
        ee.ImageCollection(collection)
        .filter(ee.Filter.eq("city_id", city_id))
        .select(RURAL_FRACTION_BAND)
        .mosaic()
    )


def get_missing_rural_city_fraction_assets(
    config: dict[str, Any],
    city_ids: list[int | str],
    halos: ee.FeatureCollection,
    projection: ee.Projection,
    existing_names: set[str]
) -> list[str]:
    """List missing direct or chunked fraction assets for selected cities."""
    export_config = config["export"]
    chunked_city_ids = set(
        export_config["rural_city_fraction_chunked_ids"]
    )
    chunk_pixels = export_config["rural_city_fraction_chunk_pixels"]
    id_field = config["cities"]["id_field"]
    missing: list[str] = []

    for city_id in city_ids:
        direct_asset = rural_city_fraction_asset_id(config, city_id)
        direct_name = direct_asset.rsplit("/", 1)[-1]
        if city_id not in chunked_city_ids or direct_name in existing_names:
            if direct_name not in existing_names:
                missing.append(direct_asset)
            continue

        city_halos = halos.filter(ee.Filter.eq(id_field, city_id))
        subtile_ids = get_occupied_subtile_ids(
            config=config,
            projection=projection,
            rural_halos=city_halos,
            subtile_pixels=chunk_pixels
        )
        if not subtile_ids:
            raise ValueError(
                f"No MODIS subtiles intersect city {city_id}'s rural halo."
            )
        for subtile_id in subtile_ids:
            chunk_asset = rural_city_fraction_chunk_asset_id(
                config,
                city_id,
                subtile_id
            )
            if chunk_asset.rsplit("/", 1)[-1] not in existing_names:
                missing.append(chunk_asset)

    return missing


def resolve_rural_city_ids(
    config: dict[str, Any],
    halos: ee.FeatureCollection
) -> list[int | str]:
    """Resolve configured city IDs against the non-empty halo asset."""
    id_field = config["cities"]["id_field"]
    available_ids = (
        halos
        .aggregate_array(id_field)
        .distinct()
        .sort()
        .getInfo()
    )
    if not available_ids:
        raise ValueError("The halo asset contains no non-empty city halos.")

    requested_ids = config["export"]["rural_city_ids"]
    if requested_ids is None:
        return available_ids
    if (
        not isinstance(requested_ids, list)
        or not requested_ids
        or any(type(city_id) not in (int, str) for city_id in requested_ids)
        or len(set(requested_ids)) != len(requested_ids)
    ):
        raise ValueError(
            "export.rural_city_ids must be null or a non-empty list of "
            "unique, non-null city IDs."
        )

    available_set = set(available_ids)
    missing_ids = [
        city_id
        for city_id in requested_ids
        if city_id not in available_set
    ]
    if missing_ids:
        raise ValueError(
            f"Configured city IDs are missing from the halo asset: "
            f"{missing_ids}"
        )

    requested_set = set(requested_ids)
    return [
        city_id
        for city_id in available_ids
        if city_id in requested_set
    ]


def get_priority_city_ids(
    config: dict[str, Any]
) -> list[int | str]:
    """Get city IDs whose polygons intersect the configured priority points."""
    points = ee.FeatureCollection(
        config["cities"]["priority_points_asset"]
    )
    polygons = ee.FeatureCollection(config["cities"]["asset"])
    id_field = config["cities"]["id_field"]
    city_ids = (
        polygons
        .filterBounds(points.geometry())
        .aggregate_array(id_field)
        .distinct()
        .sort()
        .getInfo()
    )
    if not city_ids:
        raise ValueError(
            "No city polygons intersect the configured priority points."
        )
    return city_ids


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
        .copyProperties(feature, [city_id_field])
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


def build_rural_fraction_for_region(
    config: dict[str, Any],
    modis_projection: ee.Projection,
    region: ee.Geometry,
    scale: float
) -> ee.Image:
    """Aggregate fine suitability directly to the MODIS grid."""
    rural_config = config["rural"]
    static_config = rural_config["static_assets"]
    max_error_m = rural_config["max_error_m"]
    processing_region = region.buffer(
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
    modis_fraction = (
        fine_eligibility
        .toFloat()
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=static_config[
                "max_source_pixels_per_modis_pixel"
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
        .clip(region)
    )

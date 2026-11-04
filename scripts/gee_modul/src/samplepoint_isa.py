from typing import Any

import ee

from src.samplepoint_attributes import (
    _join_by_point_and_city,
    _sample_modis_grid_image
)
from src.samplepoint_grid import (
    sample_modis_pixel_area,
    validate_max_pixels
)
from src.samplepoint_utils import samplepoint_base_fields


def build_samplepoint_isa_selectors(
    config: dict[str, Any],
    years: tuple[int, ...]
) -> list[str]:
    fields = samplepoint_base_fields(config)
    fields.extend(f"isa_urban_area_m2_{year}" for year in years)
    fields.extend(f"isa_urban_fraction_{year}" for year in years)
    if len(fields) != len(set(fields)):
        raise ValueError("Sample-point ISA selectors contain duplicates.")
    return fields


def build_fullcell_isa_image(
    isa_image: ee.Image,
    years: tuple[int, ...],
    max_pixels: int
) -> ee.Image:
    """Build annual urban indicators for weighted MODIS-cell fractions."""
    if (
        not years
        or any(type(year) is not int or not 1985 <= year <= 2021 for year in years)
        or list(years) != sorted(set(years))
    ):
        raise ValueError("ISA years must be unique, increasing and within 1985-2021.")
    validate_max_pixels(max_pixels)
    bands = [
        isa_image.gte(2023 - year)
        .And(isa_image.lte(38))
        .unmask(0, sameFootprint=False)
        .rename(f"isa_urban_fraction_{year}")
        for year in years
    ]
    return ee.Image.cat(bands)


def sample_fullcell_isa(
    fraction_image: ee.Image,
    points: ee.FeatureCollection,
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float,
    years: tuple[int, ...]
) -> ee.FeatureCollection:
    """Aggregate annual indicators and return urban area and fraction fields."""
    base_fields = samplepoint_base_fields(config)
    id_field = config["cities"]["id_field"]
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    tile_scale = config["processing"]["tile_scale"]
    fractions = _sample_modis_grid_image(
        image=fraction_image,
        reducer=ee.Reducer.mean(),
        points=points,
        properties=base_fields,
        projection=projection,
        scale=scale,
        max_pixels=max_pixels,
        tile_scale=tile_scale,
        fill_value=0
    )
    modis_areas = sample_modis_pixel_area(
        points, base_fields, projection, scale, tile_scale
    )
    joined = _join_by_point_and_city(
        primary=fractions,
        secondary=modis_areas,
        id_field=id_field,
        match_property="_modis_area"
    )

    def format_feature(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        area_feature = ee.Feature(feature.get("_modis_area"))
        cell_area = ee.Number(area_feature.get("modis_area_m2"))
        isa_properties = {}
        for year in years:
            fraction = ee.Number(feature.get(f"isa_urban_fraction_{year}"))
            area = fraction.multiply(cell_area)
            isa_properties[f"isa_urban_area_m2_{year}"] = area
            isa_properties[f"isa_urban_fraction_{year}"] = fraction
        properties = (
            ee.Dictionary(feature.toDictionary(base_fields))
            .combine(ee.Dictionary(isa_properties), overwrite=True)
        )
        return ee.Feature(None, properties)

    return joined.map(format_feature)

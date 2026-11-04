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
    fields.extend(f"isa_urban_fraction_{year}" for year in years)
    if len(fields) != len(set(fields)):
        raise ValueError("Sample-point ISA selectors contain duplicates.")
    return fields


def build_fullcell_isa_image(
    isa_image: ee.Image,
    years: tuple[int, ...],
    max_pixels: int
) -> ee.Image:
    """Build annual urban-area numerator bands for MODIS-cell reductions."""
    if (
        not years
        or any(type(year) is not int or not 1985 <= year <= 2021 for year in years)
        or list(years) != sorted(set(years))
    ):
        raise ValueError("ISA years must be unique, increasing and within 1985-2021.")
    validate_max_pixels(max_pixels)
    pixel_area = ee.Image.pixelArea().setDefaultProjection(
        isa_image.projection()
    )
    bands = [
        isa_image.gte(2023 - year)
        .And(isa_image.lte(38))
        .unmask(0, sameFootprint=False)
        .multiply(pixel_area)
        .rename(f"isa_area_m2_{year}")
        for year in years
    ]
    return ee.Image.cat(bands)


def sample_fullcell_isa(
    area_image: ee.Image,
    points: ee.FeatureCollection,
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float,
    years: tuple[int, ...]
) -> ee.FeatureCollection:
    """Reduce urban area within each MODIS cell and return cell fractions."""
    base_fields = samplepoint_base_fields(config)
    id_field = config["cities"]["id_field"]
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    tile_scale = config["processing"]["tile_scale"]
    areas = _sample_modis_grid_image(
        image=area_image,
        reducer=ee.Reducer.sum(),
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
        primary=areas,
        secondary=modis_areas,
        id_field=id_field,
        match_property="_modis_area"
    )

    def format_feature(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        area_feature = ee.Feature(feature.get("_modis_area"))
        cell_area = ee.Number(area_feature.get("modis_area_m2"))
        fractions = ee.Dictionary({
            f"isa_urban_fraction_{year}": (
                ee.Number(feature.get(f"isa_area_m2_{year}")).divide(cell_area)
            )
            for year in years
        })
        properties = (
            ee.Dictionary(feature.toDictionary(base_fields))
            .combine(fractions, overwrite=True)
        )
        return ee.Feature(None, properties)

    return joined.map(format_feature)

from typing import Any

import ee

from src.samplepoint_grid import (
    build_modis_cell_regions,
    reduce_over_modis_cells,
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
    bands = [
        isa_image.gte(2023 - year)
        .And(isa_image.lte(38))
        .unmask(0, sameFootprint=False)
        .multiply(ee.Image.pixelArea())
        .rename(f"_isa_area_{year}")
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
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    regions = build_modis_cell_regions(points, projection, scale)
    areas = reduce_over_modis_cells(
        area_image,
        regions,
        ee.Reducer.sum(),
        max_pixels,
        config["processing"]["tile_scale"]
    )

    def format_feature(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        cell_area = ee.Number(feature.get("_modis_area_m2"))
        fractions = ee.Dictionary({
            f"isa_urban_fraction_{year}": (
                ee.Number(feature.get(f"_isa_area_{year}")).divide(cell_area)
            )
            for year in years
        })
        properties = (
            ee.Dictionary(feature.toDictionary(base_fields))
            .combine(fractions, overwrite=True)
        )
        return ee.Feature(None, properties)

    return areas.map(format_feature)

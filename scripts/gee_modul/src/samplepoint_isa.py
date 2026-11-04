from typing import Any

import ee

from src.samplepoint_grid import mean_fullcell_zero_mask, validate_max_pixels
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
    """Aggregate annual ISA indicators over the entire MODIS pixel area."""
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
        .rename(f"isa_urban_fraction_{year}")
        for year in years
    ]
    return mean_fullcell_zero_mask(ee.Image.cat(bands), max_pixels)


def sample_fullcell_isa(
    fraction_image: ee.Image,
    points: ee.FeatureCollection,
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> ee.FeatureCollection:
    """Sample full-cell annual ISA fractions at existing MODIS points."""
    base_fields = samplepoint_base_fields(config)
    return fraction_image.sampleRegions(
        collection=points,
        properties=base_fields,
        projection=projection,
        scale=scale,
        tileScale=config["processing"]["tile_scale"],
        geometries=False
    )

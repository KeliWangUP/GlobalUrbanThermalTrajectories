from dataclasses import dataclass
from typing import Any

import ee

from src.samplepoint_attributes import (
    SamplePointSources,
    _join_by_point_and_city,
    _sample_modis_grid_image
)
from src.samplepoint_grid import mean_fullcell_zero_mask
from src.samplepoint_isa import build_fullcell_isa_image
from src.samplepoint_utils import samplepoint_base_fields


@dataclass(frozen=True)
class FullCellDevelopmentImages:
    isa_fractions: ee.Image
    water_fractions: ee.Image
    ghsl_image: ee.Image
    isa_years: tuple[int, ...]


def build_fullcell_water_image(water_image: ee.Image, max_pixels: int) -> ee.Image:
    """Build full-cell water and valid-WBM coverage fractions."""
    water = water_image.rename("_water_fraction_full")
    valid = water_image.mask().gt(0).rename("_water_valid_fraction_full")
    return mean_fullcell_zero_mask(ee.Image.cat([water, valid]), max_pixels)


def prepare_fullcell_development_images(
    config: dict[str, Any],
    sources: SamplePointSources
) -> FullCellDevelopmentImages:
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    return FullCellDevelopmentImages(
        isa_fractions=build_fullcell_isa_image(
            sources.isa_image, sources.isa_years, max_pixels
        ),
        water_fractions=build_fullcell_water_image(sources.water_image, max_pixels),
        ghsl_image=sources.ghsl_image,
        isa_years=sources.isa_years
    )


def sample_fullcell_development(
    points: ee.FeatureCollection,
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float,
    images: FullCellDevelopmentImages
) -> ee.FeatureCollection:
    """Sample full-cell ISA/water and legacy-compatible GHSL statistics."""
    base_fields = samplepoint_base_fields(config)
    id_field = config["cities"]["id_field"]
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    tile_scale = config["processing"]["tile_scale"]
    fractions = ee.Image.cat([
        images.isa_fractions,
        images.water_fractions,
        ee.Image.pixelArea().rename("_modis_area_m2")
    ])
    primary = fractions.sampleRegions(
        collection=points,
        properties=base_fields,
        projection=projection,
        scale=scale,
        tileScale=tile_scale,
        geometries=False
    )
    ghsl_reducer = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.count(),
        sharedInputs=True
    )
    ghsl_stats = _sample_modis_grid_image(
        image=images.ghsl_image,
        reducer=ghsl_reducer,
        points=points,
        properties=base_fields,
        projection=projection,
        scale=scale,
        max_pixels=max_pixels,
        tile_scale=tile_scale,
        fill_value=-1
    )
    joined = _join_by_point_and_city(
        primary=primary,
        secondary=ghsl_stats,
        id_field=id_field,
        match_property="_ghsl_stats",
        outer=True
    )
    isa_fields = [f"isa_urban_fraction_{year}" for year in images.isa_years]

    def format_feature(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        cell_area = ee.Number(feature.get("_modis_area_m2"))
        water_fraction = ee.Number(feature.get("_water_fraction_full"))
        valid_fraction = ee.Number(feature.get("_water_valid_fraction_full"))
        water_properties = ee.Dictionary({
            "water_area_m2": water_fraction.multiply(cell_area),
            "water_valid_area_m2": valid_fraction.multiply(cell_area),
            "water_fraction": water_fraction
        })
        match = feature.get("_ghsl_stats")
        ghsl_properties = ee.Dictionary(ee.Algorithms.If(
            ee.Algorithms.IsEqual(match, None),
            ee.Dictionary({}),
            ee.Feature(match).toDictionary()
        ))
        result_ghsl: dict[str, Any] = {}
        for year in config["ghsl_built"]["years"]:
            for measure, unit in (("surface", "m2"), ("volume", "m3")):
                name = f"built_{measure}_{year}_{unit}"
                mean_field, count_field = f"{name}_mean", f"{name}_count"
                count = ee.Number(ghsl_properties.get(count_field, -1))
                has_pixels = count.gt(0)
                result_ghsl[mean_field] = ee.Algorithms.If(
                    has_pixels, ghsl_properties.get(mean_field, None), None
                )
                result_ghsl[count_field] = ee.Algorithms.If(has_pixels, count, 0)
        output = (
            ee.Dictionary(feature.toDictionary(base_fields + isa_fields))
            .combine(water_properties, overwrite=True)
            .combine(ee.Dictionary(result_ghsl), overwrite=True)
        )
        return ee.Feature(None, output)

    return joined.map(format_feature)

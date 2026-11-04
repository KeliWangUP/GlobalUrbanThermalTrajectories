from dataclasses import dataclass
from typing import Any

import ee

from src.samplepoint_attributes import (
    SamplePointSources,
    build_samplepoint_selectors,
    _join_by_point_and_city,
    _sample_modis_grid_image
)
from src.samplepoint_grid import sample_modis_pixel_area
from src.samplepoint_isa import build_fullcell_isa_image
from src.samplepoint_utils import samplepoint_base_fields


@dataclass(frozen=True)
class FullCellDevelopmentImages:
    isa_fraction_image: ee.Image
    water_fraction_image: ee.Image
    ghsl_image: ee.Image
    isa_years: tuple[int, ...]


def build_fullcell_development_selectors(config: dict[str, Any]) -> list[str]:
    """Include annual ISA areas alongside the legacy fraction fields."""
    selectors = build_samplepoint_selectors(config)
    base_count = len(samplepoint_base_fields(config))
    isa_years = config["samplepoint_attributes"]["isa_years"]
    insert_at = base_count + len(isa_years)
    area_fields = [f"isa_urban_area_m2_{year}" for year in isa_years]
    return selectors[:insert_at] + area_fields + selectors[insert_at:]


def build_fullcell_water_image(water_image: ee.Image) -> ee.Image:
    """Build water and valid-coverage indicators on the WBM source grid."""
    water_fraction = water_image.unmask(
        0, sameFootprint=False
    ).rename("water_fraction")
    valid_fraction = (
        water_image.mask()
        .gt(0)
        .unmask(0, sameFootprint=False)
        .rename("water_valid_fraction")
    )
    return ee.Image.cat([water_fraction, valid_fraction])


def prepare_fullcell_development_images(
    config: dict[str, Any],
    sources: SamplePointSources
) -> FullCellDevelopmentImages:
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    return FullCellDevelopmentImages(
        isa_fraction_image=build_fullcell_isa_image(
            sources.isa_image, sources.isa_years, max_pixels
        ),
        water_fraction_image=build_fullcell_water_image(sources.water_image),
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
    """Aggregate indicators to the reference grid and scale fractions to areas."""
    base_fields = samplepoint_base_fields(config)
    id_field = config["cities"]["id_field"]
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    tile_scale = config["processing"]["tile_scale"]
    isa_stats = _sample_modis_grid_image(
        image=images.isa_fraction_image,
        reducer=ee.Reducer.mean(),
        points=points,
        properties=base_fields,
        projection=projection,
        scale=scale,
        max_pixels=max_pixels,
        tile_scale=tile_scale,
        fill_value=0
    )
    water_stats = _sample_modis_grid_image(
        image=images.water_fraction_image,
        reducer=ee.Reducer.mean(),
        points=points,
        properties=base_fields,
        projection=projection,
        scale=scale,
        max_pixels=max_pixels,
        tile_scale=tile_scale,
        fill_value=0
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
    modis_area_stats = sample_modis_pixel_area(
        points, base_fields, projection, scale, tile_scale
    )
    isa_water = _join_by_point_and_city(
        primary=isa_stats,
        secondary=water_stats,
        id_field=id_field,
        match_property="_water_stats"
    )
    isa_water_ghsl = _join_by_point_and_city(
        primary=isa_water,
        secondary=ghsl_stats,
        id_field=id_field,
        match_property="_ghsl_stats",
        outer=True
    )
    joined = _join_by_point_and_city(
        primary=isa_water_ghsl,
        secondary=modis_area_stats,
        id_field=id_field,
        match_property="_modis_area"
    )

    def format_feature(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        area_feature = ee.Feature(feature.get("_modis_area"))
        cell_area = ee.Number(area_feature.get("modis_area_m2"))
        isa_properties: dict[str, Any] = {}
        for year in images.isa_years:
            fraction = ee.Number(feature.get(f"isa_urban_fraction_{year}"))
            area = fraction.multiply(cell_area)
            isa_properties[f"isa_urban_area_m2_{year}"] = area
            isa_properties[f"isa_urban_fraction_{year}"] = fraction
        water_feature = ee.Feature(feature.get("_water_stats"))
        water_properties = ee.Dictionary(water_feature.toDictionary([
            "water_fraction",
            "water_valid_fraction"
        ]))
        water_fraction = ee.Number(water_properties.get("water_fraction", 0))
        valid_fraction = ee.Number(water_properties.get("water_valid_fraction", 0))
        water_area = water_fraction.multiply(cell_area)
        valid_area = valid_fraction.multiply(cell_area)
        water_output = ee.Dictionary({
            "water_area_m2": water_area,
            "water_valid_area_m2": valid_area,
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
            ee.Dictionary(feature.toDictionary(base_fields))
            .combine(ee.Dictionary(isa_properties), overwrite=True)
            .combine(water_output, overwrite=True)
            .combine(ee.Dictionary(result_ghsl), overwrite=True)
        )
        return ee.Feature(None, output)

    return joined.map(format_feature)

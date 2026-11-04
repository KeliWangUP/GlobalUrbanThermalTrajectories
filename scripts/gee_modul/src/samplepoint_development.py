from dataclasses import dataclass
from typing import Any

import ee

from src.samplepoint_attributes import (
    SamplePointSources,
    _join_by_point_and_city
)
from src.samplepoint_grid import (
    build_modis_cell_regions,
    reduce_over_modis_cells
)
from src.samplepoint_isa import build_fullcell_isa_image
from src.samplepoint_utils import samplepoint_base_fields


@dataclass(frozen=True)
class FullCellDevelopmentImages:
    isa_area_image: ee.Image
    water_area_image: ee.Image
    ghsl_image: ee.Image
    isa_years: tuple[int, ...]


def build_fullcell_water_image(water_image: ee.Image) -> ee.Image:
    """Build water and valid-coverage area bands on the WBM source grid."""
    pixel_area = ee.Image.pixelArea()
    water_area = (
        water_image.unmask(0, sameFootprint=False)
        .multiply(pixel_area)
        .rename("water_area_num_m2")
    )
    valid_area = (
        water_image.mask().gt(0)
        .unmask(0, sameFootprint=False)
        .multiply(pixel_area)
        .rename("water_valid_area_num_m2")
    )
    return ee.Image.cat([water_area, valid_area])


def prepare_fullcell_development_images(
    config: dict[str, Any],
    sources: SamplePointSources
) -> FullCellDevelopmentImages:
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    return FullCellDevelopmentImages(
        isa_area_image=build_fullcell_isa_image(
            sources.isa_image, sources.isa_years, max_pixels
        ),
        water_area_image=build_fullcell_water_image(sources.water_image),
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
    """Reduce ISA, water and GHSL directly over each MODIS pixel footprint."""
    base_fields = samplepoint_base_fields(config)
    id_field = config["cities"]["id_field"]
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    tile_scale = config["processing"]["tile_scale"]
    regions = build_modis_cell_regions(points, projection, scale)
    sum_reducer = ee.Reducer.sum()

    isa_stats = reduce_over_modis_cells(
        images.isa_area_image, regions, sum_reducer, max_pixels, tile_scale
    )
    water_stats = reduce_over_modis_cells(
        images.water_area_image, regions, sum_reducer, max_pixels, tile_scale
    )
    ghsl_reducer = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.count(),
        sharedInputs=True
    )
    ghsl_stats = reduce_over_modis_cells(
        images.ghsl_image, regions, ghsl_reducer, max_pixels, tile_scale
    )
    isa_water = _join_by_point_and_city(
        primary=isa_stats,
        secondary=water_stats,
        id_field=id_field,
        match_property="_water_stats"
    )
    joined = _join_by_point_and_city(
        primary=isa_water,
        secondary=ghsl_stats,
        id_field=id_field,
        match_property="_ghsl_stats",
        outer=True
    )

    def format_feature(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        cell_area = ee.Number(feature.get("_modis_area_m2"))
        isa_properties = ee.Dictionary({
            f"isa_urban_fraction_{year}": (
                ee.Number(feature.get(f"isa_area_m2_{year}")).divide(cell_area)
            )
            for year in images.isa_years
        })
        water_feature = ee.Feature(feature.get("_water_stats"))
        water_properties = ee.Dictionary(water_feature.toDictionary([
            "water_area_num_m2",
            "water_valid_area_num_m2"
        ]))
        water_area = ee.Number(water_properties.get("water_area_num_m2", 0))
        valid_area = ee.Number(water_properties.get("water_valid_area_num_m2", 0))
        water_output = ee.Dictionary({
            "water_area_m2": water_area,
            "water_valid_area_m2": valid_area,
            "water_fraction": water_area.divide(cell_area)
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
            .combine(isa_properties, overwrite=True)
            .combine(water_output, overwrite=True)
            .combine(ee.Dictionary(result_ghsl), overwrite=True)
        )
        return ee.Feature(None, output)

    return joined.map(format_feature)

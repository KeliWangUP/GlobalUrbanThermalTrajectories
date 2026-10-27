from math import isfinite
from typing import Any

import ee

from src.export_utils import (
    export_table_to_drive,
    initialize_gee,
    load_config
)


def _validate_config(config: dict[str, Any]) -> tuple[list[int], list[str]]:
    ghsl_config = config["ghsl_built"]
    years = ghsl_config["years"]
    if (
        not isinstance(years, list)
        or not years
        or any(type(year) is not int for year in years)
        or years != sorted(set(years))
    ):
        raise ValueError(
            "ghsl_built.years must be a non-empty, strictly increasing "
            "list of unique years."
        )
    if years != list(range(2000, 2021, 5)):
        raise ValueError(
            "ghsl_built.years must contain every 5-year interval from "
            "2000 through 2020."
        )
    if type(ghsl_config["scale_m"]) is not int or ghsl_config["scale_m"] <= 0:
        raise ValueError("ghsl_built.scale_m must be a positive integer.")
    if (
        isinstance(ghsl_config["area_max_error_m"], bool)
        or not isinstance(ghsl_config["area_max_error_m"], (int, float))
        or not isfinite(ghsl_config["area_max_error_m"])
        or ghsl_config["area_max_error_m"] <= 0
    ):
        raise ValueError(
            "ghsl_built.area_max_error_m must be a positive number."
        )

    fields = [config["cities"]["id_field"], "city_polygon_area_m2"]
    for year in years:
        fields.extend([
            f"built_surface_sum_{year}_m2",
            f"built_surface_pixel_count_{year}",
            f"built_volume_total_sum_{year}_m3",
            f"built_volume_total_pixel_count_{year}"
        ])
    if len(fields) != len(set(fields)):
        raise ValueError("GHSL export selectors contain duplicate field names.")
    return years, fields


def build_city_statistics(
    config: dict[str, Any]
) -> tuple[ee.FeatureCollection, list[str]]:
    """Reduce GHSL built surface and volume over each configured city."""
    years, selectors = _validate_config(config)
    city_config = config["cities"]
    ghsl_config = config["ghsl_built"]
    id_field = city_config["id_field"]
    cities = ee.FeatureCollection(city_config["asset"])

    sum_bands = []
    count_bands = []
    sum_fields = []
    count_fields = []
    for year in years:
        surface = ee.Image(
            f"{ghsl_config['built_surface_collection']}/{year}"
        ).select(ghsl_config["built_surface_band"])
        volume = ee.Image(
            f"{ghsl_config['built_volume_collection']}/{year}"
        ).select(ghsl_config["built_volume_band"])

        surface_sum_field = f"built_surface_sum_{year}_m2"
        surface_count_field = f"built_surface_pixel_count_{year}"
        volume_sum_field = f"built_volume_total_sum_{year}_m3"
        volume_count_field = f"built_volume_total_pixel_count_{year}"

        sum_bands.extend([
            surface.rename(surface_sum_field),
            volume.rename(volume_sum_field)
        ])
        count_bands.extend([
            surface.rename(surface_count_field),
            volume.rename(volume_count_field)
        ])
        sum_fields.extend([surface_sum_field, volume_sum_field])
        count_fields.extend([surface_count_field, volume_count_field])

    area_error = ee.ErrorMargin(ghsl_config["area_max_error_m"])

    def _add_polygon_area(feature: ee.Feature) -> ee.Feature:
        return feature.set(
            "city_polygon_area_m2",
            feature.geometry().area(maxError=area_error)
        )

    regions = cities.map(_add_polygon_area)
    scale = ghsl_config["scale_m"]
    tile_scale = config["processing"]["tile_scale"]
    summed = ee.Image.cat(sum_bands).reduceRegions(
        collection=regions,
        reducer=ee.Reducer.sum(),
        scale=scale,
        tileScale=tile_scale
    )
    counted = ee.Image.cat(count_bands).reduceRegions(
        collection=regions,
        reducer=ee.Reducer.count(),
        scale=scale,
        tileScale=tile_scale
    )

    joined = ee.FeatureCollection(
        ee.Join.saveFirst("_pixel_count_feature").apply(
            primary=summed,
            secondary=counted,
            condition=ee.Filter.equals(
                leftField=id_field,
                rightField=id_field
            )
        )
    )

    def _combine_statistics(feature: ee.Feature) -> ee.Feature:
        count_feature = ee.Feature(feature.get("_pixel_count_feature"))
        return ee.Feature(feature).set(
            count_feature.toDictionary(count_fields)
        )

    return joined.map(_combine_statistics), selectors


def export_ghsl_built_city_stats(config: dict[str, Any]) -> ee.batch.Task:
    """Submit one city-level GHSL built statistics CSV export."""
    _, selectors = _validate_config(config)
    collection, _ = build_city_statistics(config)
    export_config = config["export"]
    dataset_name = export_config["ghsl_built_dataset_name"]
    description = f"{dataset_name}_{config['project']['version']}"
    return export_table_to_drive(
        collection=collection,
        description=description,
        folder=export_config["folder"],
        file_name=description,
        selectors=selectors
    )


def main() -> None:
    config = load_config()
    initialize_gee(config)
    export_ghsl_built_city_stats(config)


if __name__ == "__main__":
    main()

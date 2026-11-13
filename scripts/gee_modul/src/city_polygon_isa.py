from math import isfinite
from typing import Any

import ee


CITY_POLYGON_AREA_FIELD = "city_polygon_area_m2"
ISA_YEARS = tuple(range(2000, 2021))


def _validate_years(years: Any) -> tuple[int, ...]:
    if not isinstance(years, (list, tuple)) or tuple(years) != ISA_YEARS:
        raise ValueError(
            "city_polygon_isa.years must list every year from 2000 through "
            "2020 in order."
        )
    return ISA_YEARS


def validate_city_polygon_isa_config(
    config: dict[str, Any]
) -> tuple[tuple[int, ...], list[str]]:
    settings = config["city_polygon_isa"]
    years = _validate_years(settings["years"])
    scale = settings["scale_m"]
    if type(scale) is not int or scale <= 0:
        raise ValueError("city_polygon_isa.scale_m must be a positive integer.")
    area_error = settings["area_max_error_m"]
    if (
        isinstance(area_error, bool)
        or not isinstance(area_error, (int, float))
        or not isfinite(area_error)
        or area_error <= 0
    ):
        raise ValueError(
            "city_polygon_isa.area_max_error_m must be a positive number."
        )
    for key in ("dataset_name", "version", "task_log"):
        value = settings[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"city_polygon_isa.{key} must be a non-empty string."
            )
    id_field = config["cities"]["id_field"]
    if not isinstance(id_field, str) or not id_field.strip():
        raise ValueError("cities.id_field must be a non-empty string.")
    return years, build_city_polygon_isa_selectors(id_field, years)


def build_city_polygon_isa_selectors(
    id_field: str,
    years: tuple[int, ...]
) -> list[str]:
    years = _validate_years(years)
    if not isinstance(id_field, str) or not id_field.strip():
        raise ValueError("City ID field must be a non-empty string.")
    selectors = [
        id_field,
        CITY_POLYGON_AREA_FIELD,
        *(f"isa_urban_area_m2_{year}" for year in years),
        *(f"isa_urban_fraction_{year}" for year in years)
    ]
    if len(selectors) != len(set(selectors)):
        raise ValueError("City polygon ISA selectors contain duplicates.")
    return selectors


def build_city_polygon_isa_area_image(
    isa_image: ee.Image,
    years: tuple[int, ...]
) -> ee.Image:
    """Build annual ISA-area bands, treating masked source pixels as zero."""
    years = _validate_years(years)
    pixel_area = ee.Image.pixelArea()
    area_bands = [
        isa_image.gte(2023 - year)
        .And(isa_image.lte(38))
        .unmask(0, sameFootprint=False)
        .multiply(pixel_area)
        .rename(f"isa_urban_area_m2_{year}")
        for year in years
    ]
    return ee.Image.cat(area_bands)


def build_city_polygon_isa_statistics(
    cities: ee.FeatureCollection,
    area_image: ee.Image,
    years: tuple[int, ...],
    scale_m: int,
    area_max_error_m: float,
    tile_scale: int
) -> ee.FeatureCollection:
    years = _validate_years(years)
    area_error = ee.ErrorMargin(area_max_error_m)

    def add_polygon_area(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        area = feature.geometry().area(maxError=area_error)
        return feature.set(CITY_POLYGON_AREA_FIELD, area)

    regions = cities.map(add_polygon_area)
    area_sums = area_image.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.sum(),
        scale=scale_m,
        tileScale=tile_scale
    )

    def add_area_fractions(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        polygon_area = ee.Number(feature.get(CITY_POLYGON_AREA_FIELD))
        fractions = {
            f"isa_urban_fraction_{year}": ee.Number(
                feature.get(f"isa_urban_area_m2_{year}")
            ).divide(polygon_area)
            for year in years
        }
        return feature.set(fractions)

    return area_sums.map(add_area_fractions)

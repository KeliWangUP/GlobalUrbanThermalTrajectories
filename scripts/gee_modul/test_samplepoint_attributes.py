import json
import math

import ee

from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.samplepoint_attributes import (
    build_samplepoint_attribute_collection,
    prepare_samplepoint_sources
)


def main() -> None:
    config = load_config()
    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)

    point_asset = config["export"]["asset_root"]
    id_field = config["cities"]["id_field"]
    city_id = config["test"]["city_id"]
    city_points = ee.FeatureCollection(point_asset).filter(
        ee.Filter.eq(id_field, city_id)
    )
    city_point_count = city_points.size().getInfo()
    if city_point_count == 0:
        raise ValueError(
            f"No sample points for city {city_id} in {point_asset}."
        )

    sources = prepare_samplepoint_sources(config)
    sample = build_samplepoint_attribute_collection(
        points=city_points.limit(1),
        config=config,
        projection=projection,
        modis_scale=scale,
        sources=sources
    )
    result = ee.Feature(sample.first()).toDictionary().getInfo()
    histogram = json.loads(result["isa_value_counts"])
    fractions = json.loads(result["isa_value_fractions"])
    if not histogram:
        raise ValueError("The test sample point returned an empty ISA histogram.")
    expected_classes = {str(value) for value in sources.isa_class_values}
    if set(histogram) != expected_classes:
        raise ValueError(
            "ISA histogram must include every configured class, including "
            "classes with zero pixels."
        )
    if result["isa_valid_pixel_count"] <= 0:
        raise ValueError("The test sample point returned no valid ISA pixels.")
    if not math.isclose(
        sum(histogram.values()),
        result["isa_valid_pixel_count"],
        abs_tol=1e-6
    ):
        raise ValueError(
            "ISA class counts do not sum to isa_valid_pixel_count."
        )
    if set(histogram) != set(fractions):
        raise ValueError(
            "ISA count and fraction histograms have different class values."
        )
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-6):
        raise ValueError(
            "ISA class fractions do not sum to 1 for the test sample point."
        )

    water_fraction = result.get("water_fraction")
    if water_fraction is not None and not 0 <= water_fraction <= 1:
        raise ValueError(
            f"Water fraction is outside [0, 1]: {water_fraction}"
        )

    ghsl_years = config["ghsl_built"]["years"]
    valid_ghsl_counts = [
        result.get(f"built_surface_{year}_m2_count", 0)
        for year in ghsl_years
    ] + [
        result.get(f"built_volume_{year}_m3_count", 0)
        for year in ghsl_years
    ]
    if not any(count and count > 0 for count in valid_ghsl_counts):
        raise ValueError(
            "No valid GHSL pixels were returned for any configured year."
        )

    print(f"[PASS] City {city_id}; sample points available: {city_point_count}")
    print(f"[PASS] Sample pixel: {result.get('pixel_id')}")
    print(f"[PASS] ISA class counts: {histogram}")
    print(f"[PASS] ISA class fractions: {fractions}")
    print(f"[PASS] Water fraction: {water_fraction}")
    for year in ghsl_years:
        print(
            f"[GHSL] {year}: surface mean="
            f"{result.get(f'built_surface_{year}_m2_mean')}; volume mean="
            f"{result.get(f'built_volume_{year}_m3_mean')}; "
            f"valid pixels="
            f"{result.get(f'built_surface_{year}_m2_count')}/"
            f"{result.get(f'built_volume_{year}_m3_count')}"
        )


if __name__ == "__main__":
    main()

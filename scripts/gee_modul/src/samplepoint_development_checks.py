import copy
import math
from typing import Any

import ee

from src.samplepoint_attributes import SamplePointSources, prepare_samplepoint_sources
from src.samplepoint_development import (
    prepare_fullcell_development_images,
    sample_fullcell_development
)
from src.samplepoint_isa_checks import (
    SYNTHETIC_CRS,
    _validate_fractions,
    synthetic_fullcell_grid,
    synthetic_isa_classes
)
from src.samplepoint_priority import load_priority_city_ids_json
from src.samplepoint_grid import sample_modis_pixel_area
from src.samplepoint_utils import load_sample_points, samplepoint_base_fields


def _single_feature(info: dict[str, Any], location: str) -> dict[str, Any]:
    features = info.get("features", [])
    if len(features) != 1:
        raise ValueError(f"{location}: expected one feature, got {len(features)}.")
    return features[0]["properties"]


def _finite_number(properties: dict[str, Any], field: str, location: str) -> float:
    value = properties.get(field)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{location}: missing or non-finite {field}: {value!r}.")
    return float(value)


def _validate_development_values(
    properties: dict[str, Any],
    isa_years: tuple[int, ...],
    ghsl_years: list[int],
    cell_area: float,
    location: str
) -> None:
    _validate_fractions(properties, isa_years, location)
    if not math.isfinite(cell_area) or cell_area <= 0:
        raise ValueError(f"{location}: invalid MODIS cell area: {cell_area}.")
    tolerance = max(1e-4, cell_area * 1e-6)
    for year in isa_years:
        area = _finite_number(
            properties, f"isa_urban_area_m2_{year}", location
        )
        fraction = _finite_number(
            properties, f"isa_urban_fraction_{year}", location
        )
        if (
            area < -tolerance
            or area > cell_area + tolerance
            or abs(area - fraction * cell_area) > tolerance
        ):
            raise ValueError(
                f"{location}: inconsistent ISA area/fraction for {year}: "
                f"area={area}, fraction={fraction}, cell={cell_area}."
            )
    water_area = _finite_number(properties, "water_area_m2", location)
    valid_area = _finite_number(properties, "water_valid_area_m2", location)
    fraction = _finite_number(properties, "water_fraction", location)
    if (
        not -1e-6 <= fraction <= 1 + 1e-6
        or water_area < -tolerance
        or valid_area < -tolerance
        or water_area > valid_area + tolerance
        or valid_area > cell_area + tolerance
        or abs(water_area - fraction * cell_area) > tolerance
    ):
        raise ValueError(
            f"{location}: inconsistent full-cell water area, coverage or fraction: "
            f"water={water_area}, valid={valid_area}, fraction={fraction}, cell={cell_area}."
        )
    for year in ghsl_years:
        for measure, unit in (("surface", "m2"), ("volume", "m3")):
            prefix = f"built_{measure}_{year}_{unit}"
            count = _finite_number(properties, f"{prefix}_count", location)
            mean = properties.get(f"{prefix}_mean")
            if count < 0 or count != int(count):
                raise ValueError(f"{location}: invalid {prefix}_count: {count}.")
            if count == 0:
                if mean is not None:
                    raise ValueError(f"{location}: {prefix}_mean exists with zero count.")
            elif type(mean) not in (int, float) or not math.isfinite(mean) or mean < 0:
                raise ValueError(f"{location}: invalid {prefix}_mean: {mean!r}.")


def check_synthetic_development(config: dict[str, Any]) -> None:
    """Test full-cell water area and GHSL behavior on a tiny shifted grid."""
    target, x, y, points = synthetic_fullcell_grid(config)
    mask = x.neq(1).Or(y.neq(1))
    water = x.eq(1).And(y.eq(0)).updateMask(mask).rename("water")
    surface = x.multiply(0).add(1000).updateMask(mask).rename("built_surface_2020_m2")
    volume = x.multiply(0).add(2000).updateMask(mask).rename("built_volume_2020_m3")
    ghsl = ee.Image.cat([surface, volume])
    synthetic_config = copy.deepcopy(config)
    synthetic_config["samplepoint_attributes"]["max_pixels_per_cell"] = 16
    synthetic_config["ghsl_built"]["years"] = [2020]
    source = SamplePointSources(
        isa_image=synthetic_isa_classes(x, y),
        isa_band="isa",
        isa_years=(1985, 2020, 2021),
        ghsl_image=ghsl,
        water_image=water
    )
    area_info = sample_modis_pixel_area(
        points, ["pixel_id"], target, 60, synthetic_config["processing"]["tile_scale"]
    ).getInfo()
    cell_area = _finite_number(
        _single_feature(area_info, "Synthetic area"),
        "modis_area_m2",
        "Synthetic area"
    )
    if not math.isclose(cell_area, 3600, rel_tol=0.01):
        raise ValueError(f"Synthetic target cell area should be near 3600 m², got {cell_area}.")
    images = prepare_fullcell_development_images(synthetic_config, source)
    info = sample_fullcell_development(points, synthetic_config, target, 60, images).getInfo()
    values = _single_feature(info, "Synthetic development")
    _validate_development_values(values, source.isa_years, [2020], cell_area, "Synthetic development")
    expected = {
        "water_fraction": 0.25,
        "water_area_m2": 0.25 * cell_area,
        "water_valid_area_m2": 0.75 * cell_area,
        "built_surface_2020_m2_mean": 1000,
        "built_volume_2020_m3_mean": 2000
    }
    for field, expected_value in expected.items():
        actual = _finite_number(values, field, "Synthetic development")
        if not math.isclose(actual, expected_value, rel_tol=1e-6, abs_tol=1e-4):
            raise ValueError(f"Synthetic {field}: expected {expected_value}, got {actual}.")

    missing_ghsl = ghsl.clip(
        ee.Geometry.Rectangle(
            [1000000, 1000000, 1000060, 1000060],
            proj=ee.Projection(SYNTHETIC_CRS),
            geodesic=False
        )
    )
    missing_source = SamplePointSources(
        isa_image=source.isa_image,
        isa_band=source.isa_band,
        isa_years=source.isa_years,
        ghsl_image=missing_ghsl,
        water_image=source.water_image
    )
    missing_images = prepare_fullcell_development_images(synthetic_config, missing_source)
    missing_info = sample_fullcell_development(
        points, synthetic_config, target, 60, missing_images
    ).getInfo()
    missing_values = _single_feature(missing_info, "Synthetic missing GHSL")
    _validate_development_values(
        missing_values, source.isa_years, [2020], cell_area, "Synthetic missing GHSL"
    )
    for field in ("built_surface_2020_m2_count", "built_volume_2020_m3_count"):
        if missing_values.get(field) != 0:
            raise ValueError(f"Synthetic missing GHSL should have zero {field}.")
    missing_water_source = SamplePointSources(
        isa_image=source.isa_image,
        isa_band=source.isa_band,
        isa_years=source.isa_years,
        ghsl_image=source.ghsl_image,
        water_image=source.water_image.updateMask(ee.Image.constant(0))
    )
    missing_water_images = prepare_fullcell_development_images(
        synthetic_config, missing_water_source
    )
    missing_water_info = sample_fullcell_development(
        points, synthetic_config, target, 60, missing_water_images
    ).getInfo()
    missing_water = _single_feature(missing_water_info, "Synthetic missing WBM")
    _validate_development_values(
        missing_water, source.isa_years, [2020], cell_area, "Synthetic missing WBM"
    )
    for field in ("water_area_m2", "water_valid_area_m2", "water_fraction"):
        if missing_water.get(field) != 0:
            raise ValueError(f"Synthetic missing WBM should have zero {field}.")
    print("[PASS] Synthetic full-cell water area and GHSL means/missing coverage.")


def check_samplepoint_development_city(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> None:
    settings = config["samplepoint_development_export"]
    limit = settings["test_point_limit"]
    pixel_id = settings["test_pixel_id"]
    if type(limit) is not int or limit < 1:
        raise ValueError("samplepoint_development_export.test_point_limit must be positive.")
    if pixel_id is not None and (not isinstance(pixel_id, str) or not pixel_id.strip()):
        raise ValueError("samplepoint_development_export.test_pixel_id must be null or a non-empty string.")
    city_ids, _, _ = load_priority_city_ids_json(config)
    city_id = config["test"]["city_id"]
    if city_id not in city_ids:
        raise ValueError(f"Test city {city_id} is not in the priority-city list.")

    id_field = config["cities"]["id_field"]
    points, _ = load_sample_points(config, samplepoint_base_fields(config))
    city_points = points.filter(ee.Filter.eq(id_field, city_id))
    city_count = city_points.size().getInfo()
    if not city_count:
        raise ValueError(f"No sample points for test city {city_id}.")
    selected = city_points
    if pixel_id is not None:
        selected = selected.filter(ee.Filter.eq("pixel_id", pixel_id))
    selected = selected.sort("pixel_id").limit(limit)
    expected_ids = selected.aggregate_array("pixel_id").getInfo()
    if not expected_ids:
        raise ValueError(f"No selected points for city {city_id}, pixel={pixel_id}.")

    sources = prepare_samplepoint_sources(config)
    images = prepare_fullcell_development_images(config, sources)
    info = sample_fullcell_development(selected, config, projection, scale, images).getInfo()
    features = info.get("features", [])
    received_ids = [feature["properties"].get("pixel_id") for feature in features]
    if (
        any(not isinstance(value, str) for value in expected_ids + received_ids)
        or len(received_ids) != len(expected_ids)
        or sorted(received_ids) != sorted(expected_ids)
    ):
        raise ValueError(
            f"Sample output did not preserve selected IDs: "
            f"expected={expected_ids}, received={received_ids}."
        )
    area_info = sample_modis_pixel_area(
        selected,
        ["pixel_id", id_field],
        projection,
        scale,
        config["processing"]["tile_scale"]
    ).getInfo()
    areas: dict[str, float] = {}
    for feature in area_info.get("features", []):
        properties = feature["properties"]
        if properties.get(id_field) != city_id:
            raise ValueError(f"Unexpected city ID in cell-area output: {properties.get(id_field)}.")
        name = properties["pixel_id"]
        if name in areas:
            raise ValueError(f"Duplicate pixel ID in cell-area output: {name}.")
        areas[name] = _finite_number(
            properties, "modis_area_m2", f"Pixel {name}"
        )
    if set(areas) != set(expected_ids):
        raise ValueError("Cell-area output does not match the selected point IDs.")

    for feature in features:
        properties = feature["properties"]
        if properties.get(id_field) != city_id:
            raise ValueError(f"Unexpected city ID in output: {properties.get(id_field)}.")
        name = properties["pixel_id"]
        _validate_development_values(
            properties, sources.isa_years, config["ghsl_built"]["years"],
            areas[name], f"Pixel {name}"
        )
        print(
            f"[RESULT] pixel={name}; "
            f"ISA_2021_area_m2={properties['isa_urban_area_m2_2021']}; "
            f"ISA_2021_fraction={properties['isa_urban_fraction_2021']}; "
            f"water_fraction={properties['water_fraction']}; "
            f"water_area_m2={properties['water_area_m2']}; "
            f"valid_water_area_m2={properties['water_valid_area_m2']}; "
            f"MODIS_area_m2={areas[name]}."
        )
    print(f"[PASS] City {city_id}: {len(features)} checked sample points out of {city_count}.")

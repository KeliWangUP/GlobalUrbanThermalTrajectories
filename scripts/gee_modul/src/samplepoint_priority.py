import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import ee

from src.export_utils import CONFIG_PATH


def _city_ids_digest(city_ids: list[int]) -> str:
    id_bytes = ",".join(str(city_id) for city_id in city_ids).encode("ascii")
    return hashlib.sha256(id_bytes).hexdigest()


def load_priority_city_ids(
    config: dict[str, Any],
    id_field: str
) -> tuple[list[int], str, Path]:
    """Read the actual historical priority-city IDs from a CSV snapshot."""
    csv_path = Path(config["cities"]["priority_city_csv"])
    if not csv_path.is_absolute():
        csv_path = CONFIG_PATH.parent / csv_path
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Priority-city CSV not found: {csv_path}")

    city_ids = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or id_field not in reader.fieldnames:
            raise ValueError(f"Priority-city CSV lacks '{id_field}': {csv_path}")
        for line_number, row in enumerate(reader, start=2):
            value = row.get(id_field)
            if value is None or not value.strip():
                raise ValueError(
                    f"Missing '{id_field}' at {csv_path}:{line_number}."
                )
            try:
                city_id = int(value.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid '{id_field}' at {csv_path}:{line_number}: {value!r}."
                ) from exc
            city_ids.add(city_id)
    if not city_ids:
        raise ValueError(f"Priority-city CSV has no city IDs: {csv_path}")

    sorted_ids = sorted(city_ids)
    return sorted_ids, _city_ids_digest(sorted_ids), csv_path


def load_priority_city_ids_json(
    config: dict[str, Any]
) -> tuple[list[int], str, Path]:
    """Load the historical priority-city snapshot as unique integer IDs."""
    city_config = config["cities"]
    expected_count = city_config["priority_city_expected_count"]
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("cities.priority_city_expected_count must be positive.")
    json_path = Path(city_config["priority_city_ids_json"])
    if not json_path.is_absolute():
        json_path = CONFIG_PATH.parent / json_path
    json_path = json_path.resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"Priority-city JSON not found: {json_path}")
    city_ids = json.loads(json_path.read_text(encoding="utf-8"))
    if (
        not isinstance(city_ids, list)
        or len(city_ids) != expected_count
        or any(type(city_id) is not int for city_id in city_ids)
        or len(set(city_ids)) != len(city_ids)
    ):
        raise ValueError(
            f"Priority-city JSON must contain {expected_count} unique integer IDs: "
            f"{json_path}"
        )
    sorted_ids = sorted(city_ids)
    return sorted_ids, _city_ids_digest(sorted_ids), json_path


def filter_priority_sample_points(
    points: ee.FeatureCollection,
    id_field: str,
    city_ids: list[int]
) -> tuple[ee.FeatureCollection, int]:
    """Require every requested city to have sample points before filtering."""
    available_ids = points.aggregate_array(id_field).distinct().getInfo()
    if not available_ids:
        raise ValueError("Sample-point asset contains no city IDs.")
    missing = sorted(set(city_ids) - set(available_ids))
    if missing:
        raise ValueError(
            f"{len(missing)} priority cities have no sample points in the "
            f"configured asset; first IDs: {missing[:20]}."
        )
    selected = points.filter(ee.Filter.inList(id_field, city_ids))
    selected_count = selected.size().getInfo()
    if not selected_count:
        raise ValueError("No priority-city sample points were selected.")
    return selected, selected_count


def export_priority_city_ids_json(
    config: dict[str, Any]
) -> tuple[Path, int, str, bool]:
    """Write the historical IDs as a JSON array without overwriting data."""
    city_config = config["cities"]
    expected_count = city_config["priority_city_expected_count"]
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("cities.priority_city_expected_count must be positive.")
    city_ids, digest, _ = load_priority_city_ids(config, city_config["id_field"])
    if len(city_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} priority cities, found {len(city_ids)}."
        )

    output_path = Path(city_config["priority_city_ids_json"])
    if not output_path.is_absolute():
        output_path = CONFIG_PATH.parent / output_path
    output_path = output_path.resolve()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            existing_ids = json.load(file)
        if (
            not isinstance(existing_ids, list)
            or any(type(city_id) is not int for city_id in existing_ids)
            or existing_ids != city_ids
        ):
            raise FileExistsError(
                f"Existing file differs from the priority-city list: {output_path}. "
                "Refusing to overwrite it."
            )
        return output_path, len(city_ids), digest, False

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(city_ids, file, ensure_ascii=True, indent=2)
        file.write("\n")
    return output_path, len(city_ids), digest, True

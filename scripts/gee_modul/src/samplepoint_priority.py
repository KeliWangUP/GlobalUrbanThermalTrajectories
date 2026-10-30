import csv
import hashlib
from pathlib import Path
from typing import Any

import ee

from src.export_utils import CONFIG_PATH


def load_priority_city_ids(
    config: dict[str, Any],
    id_field: str
) -> tuple[list[int], str, Path]:
    """Read the actual historical priority-city IDs from a CSV snapshot."""
    csv_path = Path(config["samplepoint_lst_spatial_export"]["priority_city_csv"])
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
    id_bytes = ",".join(str(city_id) for city_id in sorted_ids).encode("ascii")
    digest = hashlib.sha256(id_bytes).hexdigest()
    return sorted_ids, digest, csv_path


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

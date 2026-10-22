from typing import Any

import ee

from src.export_utils import asset_exists

MAX_BATCH_COUNT = 100


def get_batch_count(config: dict[str, Any]) -> int:
    """Return the configured number of city batches."""
    count = config["batches"]["count"]
    if not isinstance(count, int) or not 1 <= count <= MAX_BATCH_COUNT:
        raise ValueError(
            f"batches.count must be an integer from 1 to {MAX_BATCH_COUNT}."
        )
    return count


def get_batch_asset_id(config: dict[str, Any], batch_index: int) -> str:
    """Return the asset ID of one stored city batch."""
    batch_config = config["batches"]
    return (
        f"{batch_config['folder_asset']}/"
        f"{batch_config['asset_name_prefix']}_{batch_index:02d}"
    )


def get_city_ids(config: dict[str, Any]) -> list[Any]:
    """Return the sorted, validated city IDs of the source boundaries."""
    city_config = config["cities"]
    city_ids = (
        ee.FeatureCollection(city_config["asset"])
        .aggregate_array(city_config["id_field"])
        .getInfo()
    )
    if any(city_id is None for city_id in city_ids):
        raise ValueError(
            f"Some cities have a null {city_config['id_field']}."
        )
    if len(set(city_ids)) != len(city_ids):
        raise ValueError(
            f"{city_config['id_field']} values are not unique."
        )
    return sorted(city_ids)


def compute_batch_assignment(config: dict[str, Any]) -> list[list[Any]]:
    """
    Assign sorted city IDs to batches round-robin.

    The assignment depends only on the city ID set and batches.count, so
    it is reproducible. Batch sizes differ by at most one city.
    """
    batch_count = get_batch_count(config)
    city_ids = get_city_ids(config)
    if batch_count > len(city_ids):
        raise ValueError(
            f"Cannot split {len(city_ids)} cities into {batch_count} batches."
        )
    return [city_ids[index::batch_count] for index in range(batch_count)]


def resolve_batch_indices(config: dict[str, Any]) -> list[int]:
    """Return the batches selected for export; null selects all batches."""
    batch_count = get_batch_count(config)
    selected = config["export"].get("rural_batch_indices")
    if selected is None:
        return list(range(batch_count))

    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(
            not isinstance(index, int) or not 0 <= index < batch_count
            for index in selected
        )
    ):
        raise ValueError(
            "export.rural_batch_indices must be null or a non-empty list "
            f"of unique integers from 0 to {batch_count - 1}."
        )
    return sorted(selected)


def verify_city_batches(
    config: dict[str, Any],
    batch_indices: list[int],
    assignment: list[list[Any]] | None = None
) -> None:
    """Require stored batch assets to match the current city assignment."""
    if assignment is None:
        assignment = compute_batch_assignment(config)
    id_field = config["cities"]["id_field"]

    for batch_index in batch_indices:
        asset_id = get_batch_asset_id(config, batch_index)
        if not asset_exists(asset_id):
            raise FileNotFoundError(
                f"City batch asset not found: {asset_id}. Run "
                "build_city_batches_asset.py and wait for its tasks to "
                "complete."
            )
        stored_ids = sorted(
            ee.FeatureCollection(asset_id)
            .aggregate_array(id_field)
            .getInfo()
        )
        if stored_ids != assignment[batch_index]:
            raise ValueError(
                f"City batch asset {asset_id} does not match the current "
                "city boundaries and batches.count. Rebuild the batch "
                "assets under a new folder_asset."
            )


def load_city_batch(
    config: dict[str, Any],
    batch_index: int
) -> ee.FeatureCollection:
    """Load one stored city batch."""
    return ee.FeatureCollection(get_batch_asset_id(config, batch_index))

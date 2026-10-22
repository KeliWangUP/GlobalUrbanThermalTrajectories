import ee

from src.batch_utils import (
    compute_batch_assignment,
    get_batch_asset_id
)
from src.export_utils import (
    asset_exists,
    initialize_gee,
    load_config
)


def ensure_folder(folder_asset: str) -> None:
    """Create the destination folder asset if it does not exist."""
    if not asset_exists(folder_asset):
        ee.data.createAsset({"type": "FOLDER"}, folder_asset)
        print(f"[INFO] Created folder asset: {folder_asset}")
        return

    asset_type = ee.data.getAsset(folder_asset).get("type")
    if asset_type != "FOLDER":
        raise ValueError(
            f"Batch destination must be a FOLDER asset, got {asset_type}: "
            f"{folder_asset}"
        )


def main() -> None:
    config = load_config()
    initialize_gee(config)

    city_config = config["cities"]
    id_field = city_config["id_field"]
    cities = ee.FeatureCollection(city_config["asset"])
    assignment = compute_batch_assignment(config)
    print(
        f"[INFO] {sum(len(ids) for ids in assignment)} cities in "
        f"{len(assignment)} batches (largest: "
        f"{max(len(ids) for ids in assignment)})."
    )

    ensure_folder(config["batches"]["folder_asset"])

    submitted = 0
    for batch_index, city_ids in enumerate(assignment):
        asset_id = get_batch_asset_id(config, batch_index)
        if asset_exists(asset_id):
            print(f"[SKIP] Batch asset exists: {asset_id}")
            continue

        batch = (
            cities
            .filter(ee.Filter.inList(id_field, city_ids))
            .map(lambda feature: feature.set("batch_id", batch_index))
        )
        task = ee.batch.Export.table.toAsset(
            collection=batch,
            description=asset_id.rsplit("/", 1)[-1],
            assetId=asset_id
        )
        task.start()
        submitted += 1
        print(f"[EXPORT] {asset_id} ({len(city_ids)} cities)")
        print(f"[TASK] {task.id}")

    print(
        f"[NEXT] Submitted {submitted} batch tasks. After they complete, "
        "run test_rural_pipeline.py, then run_export.py."
    )


if __name__ == "__main__":
    main()

import ee

from src.batch_utils import (
    filter_nonempty_geometries,
    get_occupied_subtile_ids,
    get_subtile_geometry
)
from src.export_utils import (
    asset_exists,
    get_modis_reference_grid,
    initialize_gee,
    list_image_asset_names,
    load_config
)
from src.rural_reference_utils import (
    build_intermediate_modis_projection,
    build_rural_fraction_for_subtile
)


def ensure_image_collection(asset_id: str) -> None:
    """Create the destination ImageCollection asset when necessary."""
    if not asset_exists(asset_id):
        ee.data.createAsset(
            {"type": "IMAGE_COLLECTION"},
            asset_id
        )
        print(f"[INFO] Created ImageCollection: {asset_id}")
        return

    asset = ee.data.getAsset(asset_id)
    if asset.get("type") != "IMAGE_COLLECTION":
        raise ValueError(
            f"Fraction destination must be an IMAGE_COLLECTION: {asset_id}"
        )


def main() -> None:
    config = load_config()
    initialize_gee(config)

    rural_config = config["rural"]
    static_config = rural_config["static_assets"]
    halo_asset = static_config["halo_asset"]
    fraction_collection = static_config["fraction_collection_asset"]
    if not asset_exists(halo_asset):
        raise FileNotFoundError(
            f"Rural halo asset not found: {halo_asset}. Run "
            "build_rural_halos_asset.py first."
        )

    ensure_image_collection(fraction_collection)
    projection, scale = get_modis_reference_grid(config)
    projection_info = projection.getInfo()
    intermediate_projection = build_intermediate_modis_projection(
        modis_projection=projection,
        scale_divisor=static_config["intermediate_scale_divisor"]
    )
    halos = filter_nonempty_geometries(
        ee.FeatureCollection(halo_asset),
        rural_config["max_error_m"]
    )
    subtile_ids = get_occupied_subtile_ids(
        config,
        projection,
        halos
    )
    if not subtile_ids:
        raise ValueError(
            "No MODIS subtiles intersect the stored rural halos."
        )

    existing = list_image_asset_names(fraction_collection)
    missing = [
        subtile_id
        for subtile_id in subtile_ids
        if subtile_id not in existing
    ]
    max_tasks = static_config["max_asset_tasks_per_run"]
    if type(max_tasks) is not int or max_tasks < 1:
        raise ValueError(
            "rural.static_assets.max_asset_tasks_per_run must be a "
            "positive integer."
        )

    print(
        f"[INFO] {len(subtile_ids)} occupied MODIS subtiles; "
        f"{len(existing)} assets already exist; "
        f"{len(missing)} remain."
    )

    for subtile_id in missing[:max_tasks]:
        region = get_subtile_geometry(
            config,
            projection,
            subtile_id
        )
        fraction = build_rural_fraction_for_subtile(
            config=config,
            modis_projection=projection,
            intermediate_projection=intermediate_projection,
            subtile_region=region,
            scale=scale
        )
        asset_id = f"{fraction_collection}/{subtile_id}"
        task = ee.batch.Export.image.toAsset(
            image=fraction,
            description=f"rural_fraction_{config['project']['version']}_{subtile_id}",
            assetId=asset_id,
            region=region,
            crs=projection_info["crs"],
            crsTransform=projection_info["transform"],
            maxPixels=static_config["export_max_pixels"],
            pyramidingPolicy={"rural_fraction_scaled": "mean"}
        )
        task.start()
        print(f"[EXPORT] {asset_id}")
        print(f"[TASK] {task.id}")

    remaining = len(missing) - min(len(missing), max_tasks)
    print(
        f"[NEXT] Submitted at most {max_tasks} image tasks; "
        f"{remaining} unsubmitted subtile assets remain. Wait for active "
        "tasks to finish, then rerun this script until all assets exist."
    )
    if remaining == 0:
        print(
            "[NEXT] After the submitted tasks complete, run "
            "test_rural_pipeline.py, then run_export.py."
        )


if __name__ == "__main__":
    main()

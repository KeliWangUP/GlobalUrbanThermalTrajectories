import ee

from src.batch_utils import filter_nonempty_geometries
from src.export_utils import (
    asset_exists,
    get_modis_reference_grid,
    initialize_gee,
    list_image_asset_names,
    load_config
)
from src.rural_reference_utils import (
    build_rural_fraction_for_region,
    resolve_rural_city_ids,
    rural_city_fraction_asset_id
)


def ensure_image_collection(asset_id: str) -> None:
    """Create and validate the destination city fraction collection."""
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
            "City fraction destination must be an IMAGE_COLLECTION: "
            f"{asset_id}"
        )


def main() -> None:
    config = load_config()
    initialize_gee(config)

    rural_config = config["rural"]
    static_config = rural_config["static_assets"]
    halo_asset = static_config["halo_asset"]
    collection_asset = static_config["city_fraction_collection_asset"]
    if not asset_exists(halo_asset):
        raise FileNotFoundError(
            f"Rural halo asset not found: {halo_asset}. Run "
            "build_rural_halos_asset.py first."
        )

    ensure_image_collection(collection_asset)
    projection, scale = get_modis_reference_grid(config)
    projection_info = projection.getInfo()
    halos = filter_nonempty_geometries(
        ee.FeatureCollection(halo_asset),
        rural_config["max_error_m"]
    )
    city_ids = resolve_rural_city_ids(config, halos)

    export_config = config["export"]
    max_tasks = export_config["rural_city_fraction_max_tasks_per_run"]
    if type(max_tasks) is not int or max_tasks < 1:
        raise ValueError(
            "export.rural_city_fraction_max_tasks_per_run must be a "
            "positive integer."
        )

    existing = list_image_asset_names(collection_asset)
    missing = [
        city_id
        for city_id in city_ids
        if rural_city_fraction_asset_id(config, city_id)
        .rsplit("/", 1)[-1] not in existing
    ]
    selected_city_ids = missing[:max_tasks]
    print(
        f"[INFO] {len(city_ids)} selected cities; "
        f"{len(existing)} fraction assets exist; "
        f"{len(missing)} remain; submitting {len(selected_city_ids)} "
        "missing assets."
    )

    for city_id in selected_city_ids:
        regions = halos.filter(
            ee.Filter.eq(config["cities"]["id_field"], city_id)
        )
        region = ee.Feature(regions.first()).geometry()
        fraction = build_rural_fraction_for_region(
            config=config,
            modis_projection=projection,
            region=region,
            scale=scale
        ).set("city_id", city_id)
        asset_id = rural_city_fraction_asset_id(config, city_id)
        description = (
            f"rural_fraction_city_{config['project']['version']}_{city_id}"
        )
        task = ee.batch.Export.image.toAsset(
            image=fraction,
            description=description,
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

    remaining = len(missing) - len(selected_city_ids)
    print(
        f"[NEXT] Submitted {len(selected_city_ids)} city fraction tasks; "
        f"{max(remaining, 0)} remain after this batch."
    )
    if remaining > 0:
        print(
            "[NEXT] Wait for active tasks to finish, then rerun; "
            "completed assets will be skipped."
        )


if __name__ == "__main__":
    main()

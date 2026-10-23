from typing import Any

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
    build_rural_fraction_for_region,
    get_priority_city_ids,
    rural_city_fraction_chunk_asset_id,
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


def get_prioritized_city_ids(
    config: dict[str, Any],
    halos: ee.FeatureCollection
) -> tuple[list[int | str], int, int]:
    """Put cities intersecting the priority points before all other halos."""
    id_field = config["cities"]["id_field"]
    available_ids = (
        halos
        .aggregate_array(id_field)
        .distinct()
        .sort()
        .getInfo()
    )
    if not available_ids:
        raise ValueError("The halo asset contains no non-empty city halos.")

    points_asset = config["cities"]["priority_points_asset"]
    if not asset_exists(points_asset):
        raise FileNotFoundError(
            f"Priority city point asset not found: {points_asset}"
        )
    priority_ids = get_priority_city_ids(config)

    available_set = set(available_ids)
    priority_set = set(priority_ids)
    prioritized = [
        city_id
        for city_id in available_ids
        if city_id in priority_set
    ]
    remaining = [
        city_id
        for city_id in available_ids
        if city_id not in priority_set
    ]
    missing_halos = len(priority_set - available_set)
    return prioritized + remaining, len(prioritized), missing_halos


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
    city_ids, priority_count, priority_without_halo_count = (
        get_prioritized_city_ids(config, halos)
    )

    export_config = config["export"]
    priority_only = export_config["rural_city_fraction_priority_only"]
    if type(priority_only) is not bool:
        raise ValueError(
            "export.rural_city_fraction_priority_only must be a boolean."
        )
    if priority_only:
        city_ids = city_ids[:priority_count]

    max_tasks = export_config["rural_city_fraction_max_tasks_per_run"]
    if type(max_tasks) is not int or max_tasks < 1:
        raise ValueError(
            "export.rural_city_fraction_max_tasks_per_run must be a "
            "positive integer."
        )

    existing = list_image_asset_names(collection_asset)
    chunked_city_ids = export_config["rural_city_fraction_chunked_ids"]
    if (
        not isinstance(chunked_city_ids, list)
        or len(set(chunked_city_ids)) != len(chunked_city_ids)
        or any(type(city_id) not in (int, str) for city_id in chunked_city_ids)
    ):
        raise ValueError(
            "export.rural_city_fraction_chunked_ids must be a list of "
            "unique city IDs."
        )
    unknown_chunked_ids = set(chunked_city_ids) - set(city_ids)
    if unknown_chunked_ids:
        raise ValueError(
            "Chunked fraction city IDs are not in the selected halo queue: "
            f"{sorted(unknown_chunked_ids)}"
        )
    chunk_pixels = export_config["rural_city_fraction_chunk_pixels"]
    if type(chunk_pixels) is not int or chunk_pixels < 1:
        raise ValueError(
            "export.rural_city_fraction_chunk_pixels must be a positive "
            "integer."
        )

    chunked_set = set(chunked_city_ids)
    jobs: list[tuple[int | str, str | None]] = []
    for city_id in city_ids:
        direct_asset = rural_city_fraction_asset_id(config, city_id)
        direct_name = direct_asset.rsplit("/", 1)[-1]
        if city_id not in chunked_set:
            if direct_name not in existing:
                jobs.append((city_id, None))
            continue
        if direct_name in existing:
            continue

        city_halos = halos.filter(
            ee.Filter.eq(config["cities"]["id_field"], city_id)
        )
        subtile_ids = get_occupied_subtile_ids(
            config=config,
            projection=projection,
            rural_halos=city_halos,
            subtile_pixels=chunk_pixels
        )
        if not subtile_ids:
            raise ValueError(
                f"No MODIS subtiles intersect city {city_id}'s rural halo."
            )
        for subtile_id in subtile_ids:
            chunk_asset = rural_city_fraction_chunk_asset_id(
                config,
                city_id,
                subtile_id
            )
            if chunk_asset.rsplit("/", 1)[-1] not in existing:
                jobs.append((city_id, subtile_id))

    selected_jobs = jobs[:max_tasks]
    print(
        f"[INFO] {len(city_ids)} halo cities in queue; "
        f"{priority_count} priority cities have non-empty halos; "
        f"{priority_without_halo_count} priority cities have no non-empty "
        f"halo; priority-only mode={priority_only}; "
        f"{len(existing)} fraction assets exist; "
        f"{len(jobs)} fraction images remain; submitting "
        f"{len(selected_jobs)}."
    )

    for city_id, subtile_id in selected_jobs:
        regions = halos.filter(
            ee.Filter.eq(config["cities"]["id_field"], city_id)
        )
        halo_geometry = ee.Feature(regions.first()).geometry()
        if subtile_id is None:
            region = halo_geometry
            asset_id = rural_city_fraction_asset_id(config, city_id)
        else:
            subtile_geometry = get_subtile_geometry(
                config=config,
                projection=projection,
                subtile_id=subtile_id,
                subtile_pixels=chunk_pixels
            )
            region = halo_geometry.intersection(
                subtile_geometry,
                rural_config["max_error_m"]
            )
            asset_id = rural_city_fraction_chunk_asset_id(
                config,
                city_id,
                subtile_id
            )
        fraction = build_rural_fraction_for_region(
            config=config,
            modis_projection=projection,
            region=region,
            scale=scale
        ).set("city_id", city_id)
        description = (
            f"rural_fraction_city_{config['project']['version']}_{city_id}"
        )
        if subtile_id is not None:
            description = f"{description}_{subtile_id}"
            fraction = fraction.set("subtile_id", subtile_id)
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

    remaining = len(jobs) - len(selected_jobs)
    print(
        f"[NEXT] Submitted {len(selected_jobs)} city fraction tasks; "
        f"{max(remaining, 0)} remain after this batch."
    )
    if remaining > 0:
        print(
            "[NEXT] Wait for active tasks to finish, then rerun; "
            "completed assets will be skipped."
        )


if __name__ == "__main__":
    main()

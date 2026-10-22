from typing import Any

import ee

from src.batch_utils import filter_nonempty_geometries
from src.export_utils import (
    asset_exists,
    build_export_selectors,
    build_study_period_lst_statistics,
    export_table_to_drive,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.rural_reference_utils import (
    apply_rural_fraction_threshold,
    build_intermediate_modis_projection,
    build_rural_fraction_for_region
)


def resolve_city_ids(
    config: dict[str, Any],
    halos: ee.FeatureCollection
) -> list[int | str]:
    """Resolve configured city IDs against the non-empty halo asset."""
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

    requested_ids = config["export"]["rural_city_ids"]
    if requested_ids is None:
        return available_ids
    if (
        not isinstance(requested_ids, list)
        or not requested_ids
        or any(type(city_id) not in (int, str) for city_id in requested_ids)
        or len(set(requested_ids)) != len(requested_ids)
    ):
        raise ValueError(
            "export.rural_city_ids must be null or a non-empty list of "
            "unique, non-null city IDs."
        )

    available_set = set(available_ids)
    missing_ids = [
        city_id
        for city_id in requested_ids
        if city_id not in available_set
    ]
    if missing_ids:
        raise ValueError(
            f"Configured city IDs are missing from the halo asset: "
            f"{missing_ids}"
        )

    requested_set = set(requested_ids)
    return [
        city_id
        for city_id in available_ids
        if city_id in requested_set
    ]


def main() -> None:
    config = load_config()
    initialize_gee(config)

    rural_config = config["rural"]
    static_config = rural_config["static_assets"]
    export_config = config["export"]
    halo_asset = static_config["halo_asset"]
    if not asset_exists(halo_asset):
        raise FileNotFoundError(
            f"Rural halo asset not found: {halo_asset}. Run "
            "build_rural_halos_asset.py first."
        )

    projection, scale = get_modis_reference_grid(config)
    intermediate_projection = build_intermediate_modis_projection(
        modis_projection=projection,
        scale_divisor=static_config["intermediate_scale_divisor"]
    )
    halos = filter_nonempty_geometries(
        ee.FeatureCollection(halo_asset),
        rural_config["max_error_m"]
    )
    city_ids = resolve_city_ids(config, halos)

    start_index = export_config["rural_city_start_index"]
    max_tasks = export_config["rural_city_max_tasks_per_run"]
    if type(start_index) is not int or start_index < 0:
        raise ValueError(
            "export.rural_city_start_index must be a non-negative integer."
        )
    if type(max_tasks) is not int or max_tasks < 1:
        raise ValueError(
            "export.rural_city_max_tasks_per_run must be a positive integer."
        )
    if start_index >= len(city_ids):
        raise ValueError(
            f"rural_city_start_index {start_index} is outside the selected "
            f"city range (0 to {len(city_ids) - 1})."
        )

    selected_city_ids = city_ids[start_index:start_index + max_tasks]
    end_year = config["study"]["end_year"]
    start_year = config["study"]["start_year"]
    dataset_name = export_config["rural_city_dataset_name"]
    version = config["project"]["version"]
    id_field = config["cities"]["id_field"]
    tasks_submitted = 0

    print(
        f"[INFO] Selected {len(city_ids)} cities; submitting "
        f"{len(selected_city_ids)} starting at index {start_index}."
    )
    print(
        f"[INFO] Each city export includes {start_year}-{end_year}, "
        f"{(end_year - start_year + 1) * 12} monthly records."
    )

    for city_id in selected_city_ids:
        regions = halos.filter(ee.Filter.eq(id_field, city_id))
        halo = ee.Feature(regions.first()).geometry()
        fraction = build_rural_fraction_for_region(
            config=config,
            modis_projection=projection,
            intermediate_projection=intermediate_projection,
            region=halo,
            scale=scale
        )
        rural_mask = apply_rural_fraction_threshold(config, fraction)
        statistics = build_study_period_lst_statistics(
            regions=regions,
            config=config,
            projection=projection,
            scale=scale,
            id_field=id_field,
            spatial_mask=rural_mask
        )

        city_token = str(city_id)
        file_name = (
            f"{dataset_name}_{version}_{city_token}_"
            f"{start_year}_{end_year}"
        )
        print(f"\n[INFO] Building all-month rural export for city {city_id}")
        export_table_to_drive(
            collection=statistics,
            description=file_name,
            folder=export_config["folder"],
            file_name=file_name,
            selectors=build_export_selectors(id_field)
        )
        tasks_submitted += 1

    next_index = start_index + len(selected_city_ids)
    remaining = len(city_ids) - next_index
    print(
        f"\n[INFO] Submitted {tasks_submitted} all-years city tasks; "
        f"{remaining} selected cities remain."
    )
    if remaining:
        print(
            "[NEXT] Set export.rural_city_start_index to "
            f"{next_index} to submit the next city batch."
        )


if __name__ == "__main__":
    main()

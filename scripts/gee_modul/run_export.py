from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from export_city_monthly_lst import export_city
from export_rural_all_cities_annual import export_rural_all_cities
from export_samplepoint_attributes import export_samplepoint_attributes
from export_samplepoint_lst import export_samplepoint_lst
from export_samplepoint_lst_month import export_samplepoint_lst_month


EXPORTERS = {
    "city": export_city,
    "rural": export_rural_all_cities,
    "samplepoint_attributes": export_samplepoint_attributes,
    "samplepoint_lst": export_samplepoint_lst,
    "samplepoint_lst_month": export_samplepoint_lst_month
}


def main() -> None:
    config = load_config()
    targets = config["export"]["targets"]
    unknown_targets = set(targets) - EXPORTERS.keys()
    if unknown_targets:
        raise ValueError(
            f"Unsupported export targets in config: "
            f"{', '.join(sorted(unknown_targets))}"
        )
    if not targets:
        raise ValueError("Config must specify at least one export target.")
    if len(targets) != len(set(targets)):
        raise ValueError("Config export targets must not contain duplicates.")

    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)

    for target in targets:
        print(f"[INFO] Starting {target} exports.")
        EXPORTERS[target](config, projection, scale)


if __name__ == "__main__":
    main()

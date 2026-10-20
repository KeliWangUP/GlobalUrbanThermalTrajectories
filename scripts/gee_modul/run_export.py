from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from export_city_monthly_lst import export_city
from export_rural_monthly_lst import export_rural


EXPORTERS = {
    "city": export_city,
    "rural": export_rural
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

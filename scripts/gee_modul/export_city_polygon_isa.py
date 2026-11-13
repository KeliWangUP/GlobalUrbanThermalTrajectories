from src.city_polygon_isa_exports import export_city_polygon_isa
from src.export_utils import initialize_gee, load_config


def main() -> None:
    config = load_config()
    initialize_gee(config)
    export_city_polygon_isa(config)


if __name__ == "__main__":
    main()

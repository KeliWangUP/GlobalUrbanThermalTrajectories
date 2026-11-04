from src.export_utils import load_config
from src.samplepoint_priority import export_priority_city_ids_json


def main() -> None:
    config = load_config()
    output_path, count, digest, created = export_priority_city_ids_json(config)
    action = "Created" if created else "Already up to date"
    print(f"[INFO] {action}: {output_path}")
    print(f"[INFO] {config['cities']['id_field']} count: {count}")
    print(f"[INFO] Sorted city ID SHA-256: {digest}")


if __name__ == "__main__":
    main()

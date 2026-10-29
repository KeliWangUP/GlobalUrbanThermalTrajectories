import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export_utils import get_modis_reference_grid, initialize_gee, load_config
from src.samplepoint_attribute_exports import export_samplepoint_attributes
from src.samplepoint_recovery import FAILED_BATCH_FILE, read_failed_batch


def main() -> None:
    config = load_config()
    initialize_gee(config)
    batch_index, batch_count, batch_name = read_failed_batch(FAILED_BATCH_FILE)
    projection, scale = get_modis_reference_grid(config)
    tasks = export_samplepoint_attributes(
        config=config,
        projection=projection,
        scale=scale,
        selected_batch_index=batch_index,
        expected_batch_count=batch_count
    )
    if len(tasks) != 1:
        raise RuntimeError(
            f"Expected one retry task for '{batch_name}', got {len(tasks)}."
        )
    print(f"[INFO] Retried batch {batch_index:04d} of {batch_count:04d}.")


if __name__ == "__main__":
    main()

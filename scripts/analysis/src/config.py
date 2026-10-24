import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    urban_raw_dir: Path
    rural_raw_dir: Path
    urban_parquet_dir: Path
    rural_parquet_dir: Path
    monthly_suhi_file: Path
    seasonal_suhi_file: Path
    annual_suhi_file: Path
    parquet_compression: str


def _require_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Analysis config field '{name}' must be a JSON object.")
    return value


def _require_string(mapping: dict[str, object], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Analysis config field '{name}' must be a non-empty string."
        )
    return value


def _resolve_path(config_file: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_file.parent / path
    return path.resolve()


def load_analysis_config(config_path: Path | None = None) -> AnalysisConfig:
    config_file = (
        config_path or Path(__file__).resolve().parents[1] / "config.json"
    ).resolve()
    config_data = json.loads(config_file.read_text(encoding="utf-8"))
    root = _require_object(config_data, "root")
    paths = _require_object(root.get("paths"), "paths")
    parquet = _require_object(root.get("parquet"), "parquet")

    return AnalysisConfig(
        urban_raw_dir=_resolve_path(
            config_file, _require_string(paths, "urban_raw_dir")
        ),
        rural_raw_dir=_resolve_path(
            config_file, _require_string(paths, "rural_raw_dir")
        ),
        urban_parquet_dir=_resolve_path(
            config_file, _require_string(paths, "urban_parquet_dir")
        ),
        rural_parquet_dir=_resolve_path(
            config_file, _require_string(paths, "rural_parquet_dir")
        ),
        monthly_suhi_file=_resolve_path(
            config_file, _require_string(paths, "monthly_suhi_file")
        ),
        seasonal_suhi_file=_resolve_path(
            config_file, _require_string(paths, "seasonal_suhi_file")
        ),
        annual_suhi_file=_resolve_path(
            config_file, _require_string(paths, "annual_suhi_file")
        ),
        parquet_compression=_require_string(parquet, "compression")
    )

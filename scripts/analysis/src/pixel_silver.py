import csv
import hashlib
import json
import math
import re
import shutil
import tarfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.city_lst_csv import (
    COUNT_COLUMNS,
    EXPECTED_COLUMNS,
    get_yearly_csvs,
    load_clean_export
)


PIXEL_COLUMNS = ["pixel_id", *EXPECTED_COLUMNS]
PIXEL_ID_PATTERN = re.compile(r"MODIS_-?\d+_-?\d+")
MEMORY_PATTERN = re.compile(r"[1-9]\d*(?:MB|GB)", re.IGNORECASE)


@dataclass(frozen=True)
class PixelSilverConfig:
    config_path: Path
    pixel_lst_tar: Path
    pixel_development_parquet: Path
    rural_csv_dir: Path
    priority_city_ids_json: Path
    output_dir: Path
    start_year: int
    end_year: int
    expected_city_count: int
    compression: str
    batch_size: int
    duckdb_threads: int
    duckdb_memory_limit: str


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Pixel Silver config '{name}' must be an object.")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"Pixel Silver config '{name}' must be a positive integer.")
    return value


def load_pixel_silver_config(path: Path | None = None) -> PixelSilverConfig:
    config_path = (
        path or Path(__file__).resolve().parents[1] / "pixel_silver_config.json"
    ).resolve()
    root = _object(json.loads(config_path.read_text(encoding="utf-8")), "root")
    paths = _object(root.get("paths"), "paths")
    study = _object(root.get("study"), "study")
    processing = _object(root.get("processing"), "processing")

    def resolve_path(name: str) -> Path:
        value = paths.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Pixel Silver config 'paths.{name}' must be a path.")
        result = Path(value).expanduser()
        return (result if result.is_absolute() else config_path.parent / result).resolve()

    start = _positive_integer(study.get("start_year"), "study.start_year")
    end = _positive_integer(study.get("end_year"), "study.end_year")
    if start > end or end > 9999:
        raise ValueError("Pixel Silver study years must be ordered and at most 9999.")
    compression = processing.get("compression")
    if compression != "zstd":
        raise ValueError("Pixel Silver processing.compression must be 'zstd'.")
    memory = processing.get("duckdb_memory_limit")
    if not isinstance(memory, str) or MEMORY_PATTERN.fullmatch(memory) is None:
        raise ValueError("Pixel Silver duckdb_memory_limit must look like '1GB' or '512MB'.")

    return PixelSilverConfig(
        config_path=config_path,
        pixel_lst_tar=resolve_path("pixel_lst_tar"),
        pixel_development_parquet=resolve_path("pixel_development_parquet"),
        rural_csv_dir=resolve_path("rural_csv_dir"),
        priority_city_ids_json=resolve_path("priority_city_ids_json"),
        output_dir=resolve_path("output_dir"),
        start_year=start,
        end_year=end,
        expected_city_count=_positive_integer(
            study.get("expected_city_count"), "study.expected_city_count"
        ),
        compression=compression,
        batch_size=_positive_integer(processing.get("batch_size"), "processing.batch_size"),
        duckdb_threads=_positive_integer(
            processing.get("duckdb_threads"), "processing.duckdb_threads"
        ),
        duckdb_memory_limit=memory.upper()
    )


def _is_related_path(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_paths(config: PixelSilverConfig) -> None:
    files = (
        config.pixel_lst_tar,
        config.pixel_development_parquet,
        config.priority_city_ids_json
    )
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Pixel Silver input file not found: {path}")
    if not config.rural_csv_dir.is_dir():
        raise FileNotFoundError(f"Rural CSV directory not found: {config.rural_csv_dir}")
    if config.output_dir.exists():
        raise FileExistsError(
            f"Silver destination already exists; refusing to overwrite: {config.output_dir}"
        )
    for path in (*files, config.rural_csv_dir, config.config_path):
        if _is_related_path(config.output_dir, path):
            raise ValueError(f"Silver destination overlaps an input: {config.output_dir}, {path}")


def _load_city_ids(config: PixelSilverConfig) -> list[int]:
    values = json.loads(config.priority_city_ids_json.read_text(encoding="utf-8"))
    if (
        not isinstance(values, list)
        or len(values) != config.expected_city_count
        or any(type(value) is not int or value < 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "Priority city JSON must contain exactly "
            f"{config.expected_city_count} unique non-negative integer IDs."
        )
    return sorted(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in paths:
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed while hashing: {path}")
        result[str(path)] = {
            "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "sha256": digest
        }
    return result


def _assert_sources_unchanged(snapshot: dict[str, dict[str, Any]]) -> None:
    for name, metadata in snapshot.items():
        stat = Path(name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (metadata["bytes"], metadata["mtime_ns"]):
            raise RuntimeError(f"Input changed during Silver build: {name}")


def _connect(config: PixelSilverConfig) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute(f"SET threads = {config.duckdb_threads}")
    connection.execute(f"SET memory_limit = '{config.duckdb_memory_limit}'")
    connection.execute("SET temp_directory = ''")
    return connection


def _development_keys(
    path: Path,
    city_ids: list[int],
    connection: duckdb.DuckDBPyConnection,
    batch_size: int
) -> tuple[dict[int, set[str]], int]:
    source = pq.ParquetFile(path)
    required = {"pixel_id", "ID_UC_G0", "modis_x", "modis_y", "lon", "lat"}
    missing = required - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"Development Parquet lacks fields: {sorted(missing)}")
    pixel_type = source.schema_arrow.field("pixel_id").type
    city_type = source.schema_arrow.field("ID_UC_G0").type
    if not (pa.types.is_string(pixel_type) or pa.types.is_large_string(pixel_type)):
        raise ValueError(f"Development pixel_id must be a string, got {pixel_type}.")
    if not pa.types.is_integer(city_type):
        raise ValueError(f"Development ID_UC_G0 must be an integer, got {city_type}.")
    float_fields = []
    for field in source.schema_arrow:
        if field.name == "pixel_id":
            continue
        if pa.types.is_floating(field.type):
            float_fields.append(field.name)
        elif not pa.types.is_integer(field.type):
            raise ValueError(f"Unexpected development field type: {field.name}: {field.type}")
    if float_fields:
        for batch in source.iter_batches(batch_size=batch_size, columns=float_fields):
            for index, name in enumerate(batch.schema.names):
                invalid = pc.fill_null(pc.invert(pc.is_finite(batch.column(index))), False)
                if pc.any(invalid).as_py():
                    raise ValueError(f"Non-finite development value: {path}:{name}.")
    table = source.read(columns=["ID_UC_G0", "pixel_id"])
    connection.register("_development_keys", table)
    count, valid, unique, malformed = connection.execute(
        "SELECT count(*), "
        "count(*) FILTER (WHERE ID_UC_G0 IS NOT NULL AND pixel_id IS NOT NULL), "
        "count(DISTINCT (ID_UC_G0, pixel_id)), "
        "count(*) FILTER (WHERE pixel_id IS NOT NULL AND "
        "NOT regexp_full_match(pixel_id, 'MODIS_-?[0-9]+_-?[0-9]+')) "
        "FROM _development_keys"
    ).fetchone()
    if count == 0 or count != valid or count != unique or malformed:
        raise ValueError(
            f"Development keys are empty, null, duplicated or malformed: "
            f"rows={count}, valid={valid}, unique={unique}, malformed={malformed}."
        )
    matched = connection.execute(
        "SELECT ID_UC_G0, pixel_id FROM _development_keys "
        "WHERE ID_UC_G0 IN (SELECT unnest(?))",
        [city_ids]
    ).fetchall()
    connection.unregister("_development_keys")
    keys: dict[int, set[str]] = defaultdict(set)
    for city_id, pixel_id in matched:
        if type(city_id) is not int or not isinstance(pixel_id, str):
            raise ValueError("Development city and pixel IDs have unexpected types.")
        if PIXEL_ID_PATTERN.fullmatch(pixel_id) is None:
            raise ValueError(f"Invalid development pixel ID for city {city_id}: {pixel_id}")
        keys[city_id].add(pixel_id)
    missing_ids = sorted(set(city_ids) - set(keys))
    if missing_ids:
        raise ValueError(f"Priority cities lack development points: {missing_ids[:20]}")
    return dict(keys), count


def _archive_members(
    archive: tarfile.TarFile,
    city_ids: list[int]
) -> dict[int, tarfile.TarInfo]:
    members: dict[int, tarfile.TarInfo] = {}
    root_name = None
    for member in archive.getmembers():
        if member.isdir():
            continue
        if not member.isfile():
            raise ValueError(f"Unsupported archive entry: {member.name}")
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
            raise ValueError(f"Unexpected archive path: {member.name}")
        root, city_part, filename = path.parts
        match = re.fullmatch(r"city_id=(\d+)", city_part)
        if match is None or filename != "data.parquet":
            raise ValueError(f"Unexpected archive file: {member.name}")
        if root_name is None:
            root_name = root
        if root != root_name:
            raise ValueError(f"Inconsistent archive root: {member.name}")
        city_id = int(match.group(1))
        if city_id in members:
            raise ValueError(f"Duplicate city in archive: {city_id}")
        members[city_id] = member
    expected = set(city_ids)
    if set(members) != expected:
        raise ValueError(
            "Archive city IDs differ from the priority list; "
            f"missing={sorted(expected - set(members))[:20]}, "
            f"unexpected={sorted(set(members) - expected)[:20]}."
        )
    return members


def _validate_city_keys(
    file: Any,
    city_id: int,
    expected_pixels: set[str],
    config: PixelSilverConfig,
    connection: duckdb.DuckDBPyConnection
) -> int:
    parquet = pq.ParquetFile(file)
    if parquet.schema_arrow.names != PIXEL_COLUMNS:
        raise ValueError(f"Unexpected LST fields for city {city_id}: {parquet.schema_arrow.names}")
    keys = parquet.read(columns=["pixel_id", "ID_UC_G0", "year", "month"])
    connection.register("_city_keys", keys)
    count, unique, dates, invalid = connection.execute(
        "SELECT count(*), count(DISTINCT (pixel_id, year, month)), "
        "count(DISTINCT (year, month)), "
        "count(*) FILTER (WHERE pixel_id IS NULL OR ID_UC_G0 IS NULL "
        "OR ID_UC_G0 != ? OR year IS NULL OR year < ? OR year > ? "
        "OR month IS NULL OR month < 1 OR month > 12) FROM _city_keys",
        [city_id, config.start_year, config.end_year]
    ).fetchone()
    pixels = {row[0] for row in connection.execute(
        "SELECT DISTINCT pixel_id FROM _city_keys"
    ).fetchall()}
    connection.unregister("_city_keys")
    months = (config.end_year - config.start_year + 1) * 12
    if (
        count == 0
        or invalid != 0
        or unique != count
        or dates != months
        or count != len(pixels) * months
        or any(not isinstance(value, str) or PIXEL_ID_PATTERN.fullmatch(value) is None for value in pixels)
    ):
        raise ValueError(
            f"Invalid LST keys or month coverage for city {city_id}: "
            f"rows={count}, unique={unique}, months={dates}, invalid={invalid}."
        )
    if pixels != expected_pixels:
        raise ValueError(
            f"City {city_id} pixel IDs differ from development data: "
            f"LST-only={len(pixels - expected_pixels)}, "
            f"development-only={len(expected_pixels - pixels)}."
        )
    return count


def _silver_schema(pixel: bool) -> pa.Schema:
    fields = []
    for name in (PIXEL_COLUMNS if pixel else EXPECTED_COLUMNS):
        if name == "pixel_id":
            dtype = pa.string()
        elif name == "ID_UC_G0" or name in COUNT_COLUMNS:
            dtype = pa.int32()
        elif name == "year":
            dtype = pa.int16()
        elif name == "month":
            dtype = pa.int8()
        else:
            dtype = pa.float64()
        fields.append(pa.field(name, dtype))
    return pa.schema(fields)


def _normalise_batch(
    batch: pa.RecordBatch,
    schema: pa.Schema,
    location: str
) -> pa.RecordBatch:
    if batch.schema.names != schema.names:
        raise ValueError(f"Unexpected columns in {location}: {batch.schema.names}")
    result = []
    for index, field in enumerate(schema):
        original = batch.column(index)
        try:
            if field.name in COUNT_COLUMNS:
                values = pc.cast(original, pa.float64(), safe=True)
                invalid = pc.fill_null(pc.invert(pc.is_finite(values)), False)
                if pc.any(invalid).as_py():
                    raise ValueError(f"Non-finite count in {location}: {field.name}")
                if pc.any(pc.fill_null(pc.less(values, 0), False)).as_py():
                    raise ValueError(f"Negative count in {location}: {field.name}")
                values = pc.cast(values, field.type, safe=True)
            else:
                values = pc.cast(original, field.type, safe=True)
                if pa.types.is_floating(field.type):
                    invalid = pc.fill_null(pc.invert(pc.is_finite(values)), False)
                    if pc.any(invalid).as_py():
                        raise ValueError(f"Non-finite value in {location}: {field.name}")
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
            raise ValueError(f"Invalid value in {location}: {field.name}: {exc}") from exc
        if values.null_count != original.null_count:
            raise ValueError(f"Null count changed in {location}: {field.name}")
        result.append(values)
    return pa.RecordBatch.from_arrays(result, schema=schema)


def _write_batches(
    batches: Any,
    output: Path,
    schema: pa.Schema,
    compression: str,
    location: str
) -> tuple[int, dict[str, int]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Staged output already exists: {output}")
    count = 0
    nulls = dict.fromkeys(schema.names, 0)
    with pq.ParquetWriter(output, schema, compression=compression) as writer:
        for original in batches:
            batch = _normalise_batch(original, schema, location)
            writer.write_table(pa.Table.from_batches([batch]), row_group_size=batch.num_rows)
            count += batch.num_rows
            for index, name in enumerate(schema.names):
                nulls[name] += batch.column(index).null_count
    if count == 0:
        raise ValueError(f"No records to write for {location}.")
    return count, nulls


def _verify_footer(
    output: Path,
    schema: pa.Schema,
    expected_count: int,
    expected_nulls: dict[str, int]
) -> None:
    parquet = pq.ParquetFile(output)
    if not parquet.schema_arrow.equals(schema, check_metadata=False):
        raise ValueError(f"Unexpected output schema: {output}")
    if parquet.metadata.num_rows != expected_count:
        raise ValueError(f"Output row count mismatch: {output}")
    for index, name in enumerate(schema.names):
        null_count = 0
        for group in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(group).column(index).statistics
            if statistics is None or statistics.null_count is None:
                null_count += parquet.read_row_group(group, columns=[name]).column(0).null_count
            else:
                null_count += statistics.null_count
        if null_count != expected_nulls[name]:
            raise ValueError(f"Output null count differs for {output}:{name}.")


def _check_rural_source(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected rural CSV fields: {path}")
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Extra fields in {path}:{line}.")
            for name in EXPECTED_COLUMNS[3:]:
                value = row.get(name)
                if value is None:
                    raise ValueError(f"Missing field in {path}:{line}: {name}.")
                if not value.strip():
                    continue
                try:
                    number = float(value)
                except ValueError as exc:
                    raise ValueError(f"Invalid value in {path}:{line}: {name}={value!r}.") from exc
                if not math.isfinite(number):
                    raise ValueError(f"Non-finite value in {path}:{line}: {name}={value!r}.")


def _load_rural(
    path: Path,
    year: int,
    city_ids: set[int]
) -> pd.DataFrame:
    _check_rural_source(path)
    frame = load_clean_export(path, year)
    if set(frame["ID_UC_G0"].tolist()) != city_ids:
        raise ValueError(f"Rural city IDs differ from priority list: {path}")
    return frame


def build_pixel_silver(config: PixelSilverConfig) -> Path:
    """Build and validate a new city-partitioned Silver dataset."""
    started = datetime.now(timezone.utc).isoformat()
    _validate_paths(config)
    city_ids = _load_city_ids(config)
    city_set = set(city_ids)
    years = list(range(config.start_year, config.end_year + 1))
    rural_files = get_yearly_csvs(config.rural_csv_dir)
    if sorted(rural_files) != years:
        raise ValueError(
            f"Rural CSV years do not match {config.start_year}-{config.end_year}: "
            f"{sorted(rural_files)}."
        )
    print(f"[INFO] Checking {len(rural_files)} rural CSV files.", flush=True)
    for year in years:
        _load_rural(rural_files[year], year, city_set)
    source_paths = [
        config.config_path,
        config.pixel_lst_tar,
        config.pixel_development_parquet,
        config.priority_city_ids_json,
        *[rural_files[year] for year in years]
    ]
    print("[INFO] Hashing inputs for provenance.", flush=True)
    source_snapshot = _source_snapshot(source_paths)

    with _connect(config) as connection, tarfile.open(config.pixel_lst_tar, "r:") as archive:
        dev_keys, development_rows = _development_keys(
            config.pixel_development_parquet, city_ids, connection, config.batch_size
        )
        members = _archive_members(archive, city_ids)
        records: dict[int, dict[str, int]] = {}
        print(f"[INFO] Checking keys and month coverage for {len(members)} cities.", flush=True)
        for position, (city_id, member) in enumerate(members.items(), start=1):
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unable to read archive member: {member.name}")
            with stream:
                rows = _validate_city_keys(
                    stream, city_id, dev_keys[city_id], config, connection
                )
            records[city_id] = {"rows": rows, "pixels": len(dev_keys[city_id])}
            if position % 200 == 0:
                print(f"[CHECK] {position}/{len(members)} cities.", flush=True)

        config.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = config.output_dir.parent / f".{config.output_dir.name}_staging_{uuid.uuid4().hex}"
        staging.mkdir()
        print(f"[INFO] Staging output: {staging}", flush=True)
        pixel_schema = _silver_schema(pixel=True)
        rural_schema = _silver_schema(pixel=False)
        total_rows = 0
        lst_nulls: Counter[str] = Counter()
        print("[INFO] Converting city-partitioned LST Parquets.", flush=True)
        for position, (city_id, member) in enumerate(members.items(), start=1):
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unable to read archive member: {member.name}")
            output = staging / "pixel_lst" / f"city_id={city_id}" / "data.parquet"
            with stream:
                batches = pq.ParquetFile(stream).iter_batches(batch_size=config.batch_size)
                count, nulls = _write_batches(
                    batches, output, pixel_schema, config.compression, f"city {city_id}"
                )
            _verify_footer(output, pixel_schema, records[city_id]["rows"], nulls)
            output_count = _validate_city_keys(
                output, city_id, dev_keys[city_id], config, connection
            )
            if count != output_count:
                raise ValueError(f"City {city_id} output count changed after writing.")
            total_rows += count
            lst_nulls.update(nulls)
            if position % 50 == 0 or position == len(members):
                print(f"[LST] {position}/{len(members)} cities; {total_rows:,} rows.", flush=True)

        development_output = staging / "pixel_development" / "data.parquet"
        development_output.parent.mkdir(parents=True)
        with config.pixel_development_parquet.open("rb") as source, development_output.open("xb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        if _sha256(development_output) != source_snapshot[str(config.pixel_development_parquet)]["sha256"]:
            raise ValueError("Development Parquet copy does not match the source.")
        print(f"[DEVELOPMENT] {development_rows:,} unchanged rows.", flush=True)

        rural_rows = 0
        rural_nulls: Counter[str] = Counter()
        for year in years:
            source = rural_files[year]
            frame = _load_rural(source, year, city_set)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            output = staging / "rural_lst" / f"rural_{year}.parquet"
            count, nulls = _write_batches(
                table.to_batches(max_chunksize=config.batch_size),
                output, rural_schema, config.compression, str(source)
            )
            _verify_footer(output, rural_schema, len(city_ids) * 12, nulls)
            rural_rows += count
            rural_nulls.update(nulls)
            print(f"[RURAL] {year}: {count:,} rows.", flush=True)

        lst_pattern = str(staging / "pixel_lst" / "city_id=*" / "data.parquet")
        rural_pattern = str(staging / "rural_lst" / "rural_*.parquet")
        lst_count = connection.execute(
            "SELECT count(*) FROM read_parquet(?, hive_partitioning=false)",
            [lst_pattern]
        ).fetchone()[0]
        rural_count = connection.execute(
            "SELECT count(*) FROM read_parquet(?, hive_partitioning=false)",
            [rural_pattern]
        ).fetchone()[0]
        if lst_count != total_rows or rural_count != rural_rows:
            raise ValueError("DuckDB output row counts differ from the written records.")
        _assert_sources_unchanged(source_snapshot)

        manifest = {
            "format_version": 1,
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "start_year": config.start_year,
            "end_year": config.end_year,
            "city_count": len(city_ids),
            "pixel_lst_rows": total_rows,
            "priority_pixel_count": sum(len(values) for values in dev_keys.values()),
            "development_rows": development_rows,
            "rural_rows": rural_rows,
            "pixel_lst_null_counts": dict(lst_nulls),
            "rural_null_counts": dict(rural_nulls),
            "lst_partition": "city_id",
            "lst_schema": str(pixel_schema),
            "rural_schema": str(rural_schema),
            "sources": source_snapshot,
            "cities": {str(city_id): records[city_id] for city_id in sorted(records)}
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=True, indent=2)
            file.write("\n")

    if config.output_dir.exists():
        raise FileExistsError(f"Silver destination appeared during the build: {config.output_dir}")
    staging.rename(config.output_dir)
    print(f"[PASS] Published validated Silver dataset: {config.output_dir}", flush=True)
    return config.output_dir

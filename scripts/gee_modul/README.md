# Rural LST exports for all cities

The complete source-to-analysis workflow is documented in
[`../../DATA_PROCESSING_WORKFLOW.md`](../../DATA_PROCESSING_WORKFLOW.md).

To create one annual CSV containing all cities with a non-empty rural halo,
run from this directory:

```bash
python export_rural_all_cities_annual.py
```

The exporter reads `study.start_year` and `study.end_year` from `config.json`.
Set `export.rural_city_ids` to `null` to select all non-empty city halos.
Before submitting any task, it verifies that every selected city has its
direct or chunked city-fraction assets. Cities without a non-empty rural halo
cannot produce rural LST and are excluded.

The output dataset name is `export.rural_all_cities_dataset_name`; the current
configuration uses `rural_lst_all_cities`, keeping the previous
`rural_lst_over_300k` exports separate. The exporter submits one Drive CSV
task per study year, with all selected cities in each CSV.

The earlier successful 1,700-city annual flow remains available as
`export_rural_priority_cities_annual.py`. The `rural` target in
`run_export.py` now points to the all-city annual exporter.

Large-city fraction computation may still split that city's fraction image
into MODIS-aligned subtiles. This is per-city fraction chunking, not the
retired tile-batch temperature export path.

Download these annual CSVs into `data/raw/rural_all` for the analysis
pipeline. Keep the earlier priority-only CSVs in their existing
`data/raw/rural` folder; the analysis config reads only the all-city folder.

For the 2000-2024 analysis series, `study.start_year` and `study.end_year`
should be set to 2000 and 2024. Set
`export.rural_city_fraction_priority_only` to `false` before rebuilding city
fractions so the fraction builder includes non-priority cities as well.

## GHSL built-surface and volume statistics

To export built-up surface and built volume totals for each city polygon, run
from the repository root:

```bash
python scripts/gee_modul/export_ghsl_built_city_stats.py
```

The export uses the configured `cities.asset` and `cities.id_field`. For each
configured GHSL year (currently 2000, 2005, 2010, 2015, and 2020), it sums
`built_surface` from `JRC/GHSL/P2023A/GHS_BUILT_S` and
`built_volume_total` from `JRC/GHSL/P2023A/GHS_BUILT_V` over the city polygon.
It also exports the valid, unmasked pixel count for each band/year and the
polygon area in square metres. Sum units are square metres for built surface
and cubic metres for built volume. The reduction scale, area error tolerance,
source collections, bands, and years are configured in `config.json`.

The script submits one CSV task to the configured Earth Engine Drive folder.
Its configured basename is
`export.ghsl_built_dataset_name` with the project version appended. Download
the completed CSV locally when the Earth Engine task succeeds. This export
does not modify the source GHSL assets or the existing SUHI/metadata Parquets.

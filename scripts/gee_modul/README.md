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

Download these annual CSVs into `data/raw/rural_all_cities` for the analysis
pipeline. Keep the earlier priority-only CSVs in their existing
`data/raw/rural` folder; the analysis config reads only the all-city folder.

For the 2000-2024 analysis series, `study.start_year` and `study.end_year`
should be set to 2000 and 2024. Set
`export.rural_city_fraction_priority_only` to `false` before rebuilding city
fractions so the fraction builder includes non-priority cities as well.

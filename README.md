# chelsa-nex

## Overview
This repository provides a reproducible workflow for building bias‑corrected climate scenarios from the NASA NEX‑GDDP‑CMIP6 archive. The pipeline downloads or aggregates reference reanalysis data (ERA5‑Land or CHELSA‑W5E5), computes area‑weighted summaries for glacier polygons, and applies scaled distribution mapping to the CMIP6 ensemble. Outputs include daily precipitation and temperature time series that can feed MATILDA or other downstream hydrological models. NEX‑GDDP‑CMIP6 is accessed directly from Google Earth Engine (dataset description: [NASA/ORNL NEX‑GDDP‑CMIP6](https://developers.google.com/earth-engine/datasets/catalog/NASA_GDDP_CMIP6)).

The workflow is currently tailored to individual glaciers identified by the [Randolph Glacier Inventory v6](https://www.glims.org/RGI/) (RGI) and is designed to download, preprocess, and bias adjust climate data for those outlines (e.g., to force GloGEM or other glacier models). Batch jobs iterate over one or more RGI‑labeled glacier polygons, either as a single file or a directory of GeoPackages, and produce ready‑to‑use forcing for glacier modeling.

## Key components
- **`main.py`**: Batch entry point that loops over GeoPackage inputs, prepares single‑polygon extracts, and executes the full CMIP6 workflow via `ClimateScenarios`.
- **`matilda_functions.py`**: Hosts the CMIP6 downloader, bias‑correction routines, and visualization helpers used by the workflow, including generation of GloGEM‑compatible climate forcing files (`.dat`) ([GloGEM reference](https://doi.org/10.3389/feart.2015.00054)).
- **`era5l_preprocessing.py`**: Utilities to fetch ERA5‑Land daily means from Google Earth Engine (dataset description: [ECMWF/ERA5‑Land (daily raw)](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_RAW)) for each polygon (single download or parallel yearly batches).
- **`chelsa_preprocessing.py`**: Aggregates CHELSA‑W5E5 NetCDF tiles to polygon means for precipitation, temperature, and other variables. CHELSA‑W5E5 is not hosted in GEE; download the NetCDF tiles separately from the ISIMIP repository before running the aggregation (Karger et al. 2022, [CHELSA‑W5E5 DOI](https://doi.org/10.48364/ISIMIP.836809.3)).
- **`GCM_filter.py`**: Helper to limit the CMIP6 ensemble to a preferred set of global climate models.

## Prerequisites
- Python (see `chelsa_packages.txt` for suggested packages).
- Access to Google Earth Engine with a configured project for downloading ERA5‑Land and CMIP6 data.
- GeoPackage (`.gpkg`) files containing polygons with an `RGIId` column. The batch routine processes either a single GeoPackage file or every GeoPackage inside a directory; each row in the default layer is treated as one polygon. Polygon names are derived from the `RGIId` field (`RGI-<id>`), so non‑RGI identifiers are not currently supported in the batch workflow.

## Data sources
- **NEX‑GDDP‑CMIP6**: Accessed via Google Earth Engine ([dataset description](https://developers.google.com/earth-engine/datasets/catalog/NASA_GDDP_CMIP6)).
- **ERA5‑Land**: Downloaded directly from Google Earth Engine ([dataset description](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_RAW)).
- **CHELSA‑W5E5**: Download the NetCDF tiles separately from ISIMIP before running aggregation. Cite as: Dirk N. Karger, Stefan Lange, Chantal Hari, Christopher P.O. Reyer, Niklaus E. Zimmermann (2022): CHELSA‑W5E5 v1.0: W5E5 v1.0 downscaled with CHELSA v2.0. ISIMIP Repository. [DOI link](https://doi.org/10.48364/ISIMIP.836809.3).

## Configuration
All runtime settings live in `config.ini`:
- `[settings]`
  - `geopackage_dir` or `batch_dir` CLI flag: folder with input GeoPackages (or a single file).
  - `reanalysis`: choose `era5l` or `chelsa` as the reference dataset.
  - `era5l_dir` / `chelsa_dir`: where ERA5‑Land or CHELSA outputs are stored.
  - `output_dir`: destination for bias‑corrected CMIP6 CSVs and plots.
  - `gee_project`: Earth Engine project ID used for downloads.
  - `start_year` / `end_year`: time window for CMIP6 extraction.
  - `processes`: number of parallel workers for downloads and bias correction.
  - `download`, `load_backup`, `plots`, `force_preprocessing`: toggles controlling whether to fetch data, reuse cached files, render diagnostics, or re‑aggregate inputs.
- `[reanalysis]`
  - `chelsa_base_dir`, `chelsa_target_dir`, `chelsa_vars`: location and variables to aggregate when using CHELSA‑W5E5.
  - `convert_temperatures`: convert Kelvin to Celsius for reanalysis products.

Update paths and toggles to match your environment before running the workflow.

## Usage
1. Prepare your polygon GeoPackages and set the desired parameters in `config.ini`. The batch processor accepts a single GeoPackage or a directory of GeoPackages. It reads the default layer from each file and iterates over all rows, so ensure the layer you want to process is set as the default and contains an `RGIId` column.
2. Authenticate with Google Earth Engine (`earthengine authenticate`) if you have not initialized the CLI before.
3. Run the batch processor:
   ```bash
   python main.py --batch_dir /path/to/gpkg_folder
   ```
   - If `--batch_dir` is omitted, `geopackage_dir` from `config.ini` is used.
   - The script will create temporary single‑polygon GeoPackages in `/tmp`, download or aggregate reference data, and save bias‑corrected CMIP6 outputs to `output_dir`.

### Running individual preprocessing steps
- **ERA5‑Land only**:
  ```bash
  python -c "from era5l_preprocessing import ERA5LPolygonDownloader; \
  ERA5LPolygonDownloader('poly.gpkg', 'gee-project', './era5l', True).download_era5l()"
  ```
- **CHELSA‑W5E5 aggregation only**:
  ```bash
  python -c "from chelsa_preprocessing import chelsa_w5e5_agg; \
  chelsa_w5e5_agg('poly.gpkg', force=True)"
  ```

## Outputs
- `era5l_<polygon>.csv` or aggregated CHELSA CSVs: reference reanalysis time series per polygon.
- `RGI-<id>_<var>.csv`: bias‑corrected CMIP6 daily series for each variable and polygon in `output_dir`.
- `RGI-<id>_matilda.dat`: GloGEM/MATILDA‑ready climate forcing files produced during CMIP6 bias correction when enabled.
- Optional plot files illustrating bias correction performance when `plots = True`.

import pandas as pd
from tqdm import tqdm
import configparser
import glob
import xarray as xr
import geopandas as gpd
import os
import pyproj
import xagg as xa

# To avoid PROJ error:
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

# Silence xagg prints
xa.set_options(silent=True)

def chelsa_w5e5_agg(gpkg_path_or_layer, config_path="config.ini", force=False):
    config = configparser.ConfigParser()
    config.read(config_path)

    base_dir = config["reanalysis"]["chelsa_base_dir"]
    target_subdir = config.get("reanalysis", "chelsa_target_dir", fallback="aggregates")
    output_dir = os.path.join(base_dir, target_subdir)
    os.makedirs(output_dir, exist_ok=True)

    variables = [v.strip() for v in config["reanalysis"]["chelsa_vars"].split(",")]
    convert_temp = config.getboolean("reanalysis", "convert_temperatures")
    fill_value = 65535

    conversions = {
        "pr": {"factor": 86400, "offset": 0},
        "tas": {"factor": 1, "offset": -273.15 if convert_temp else 0},
        "tasmax": {"factor": 1, "offset": -273.15 if convert_temp else 0},
        "tasmin": {"factor": 1, "offset": -273.15 if convert_temp else 0},
        "rsds": {"factor": 1, "offset": 0},
    }

    polygon = gpd.read_file(gpkg_path_or_layer)
    poly_name = os.path.splitext(os.path.basename(gpkg_path_or_layer))[0]
    polygon = polygon.to_crs("EPSG:4326")

    # Check if all variable files and elevation file already exist
    existing_outputs = [os.path.join(output_dir, f"{poly_name}_{var}.csv") for var in variables]
    elevation_output = os.path.join(output_dir, f"{poly_name}_elevation.csv")
    all_exist = all(os.path.exists(f) for f in existing_outputs) and os.path.exists(elevation_output)

    if all_exist and not force:
        print(f"CHELSA output for '{poly_name}' already exists. Skipping.")
        return

    # Process orography (mean elevation)
    orog_file = sorted(glob.glob(f"{base_dir}/orog/*.nc"))[0]
    ds_orog = xr.open_dataset(orog_file)
    weightmap_orog = xa.pixel_overlaps(ds_orog, polygon)
    agg_orog = xa.aggregate(ds_orog["orog"], weightmap_orog)
    mean_elevation = agg_orog.to_dataframe()['orog'].values[0]
    print(f"Area-weighted mean elevation: {mean_elevation:.2f} m")
    df_orog = pd.DataFrame({"polygon": [poly_name], "elevation": [round(mean_elevation, 2)]})
    df_orog.to_csv(elevation_output, index=False)
    ds_orog.close()

    for var in variables:
        print(f"\n ** Processing variable: {var}")
        files = sorted(glob.glob(f"{base_dir}/{var}/*.nc"))
        if not files:
            print(f"** No files found for {var}, skipping.")
            continue

        results = []
        previous_grid = None
        weightmap = None

        for file in tqdm(files, desc=f"\u2192 {var}"):
            ds = xr.open_dataset(file)

            ds[var] = ds[var] * conversions[var]["factor"] + conversions[var]["offset"]
            ds[var] = ds[var].where(ds[var] != fill_value)

            current_grid = (tuple(ds["lat"].values), tuple(ds["lon"].values))

            if current_grid != previous_grid:
                try:
                    weightmap = xa.pixel_overlaps(ds, polygon)
                    previous_grid = current_grid
                except Exception as e:
                    print(f"\nFailed to generate weight map for {file}: {e}")
                    ds.close()
                    continue

            try:
                agg = xa.aggregate(ds[var], weightmap)
                df = agg.to_dataframe().reset_index()[["time", var]]
                df = round(df, 4)
                results.append(df)
            except Exception as e:
                print(f"\nFailed to aggregate {file}: {e}")

            ds.close()

        final_df = pd.concat(results)
        output_file = os.path.join(output_dir, f"{poly_name}_{var}.csv")
        final_df.to_csv(output_file, index=False)
        print(f"** Saved: {output_file}")

    print("\n** Done.")

# Example usage
# chelsa_w5e5_agg("/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/cholpon_ata/gpkg_glaciers/cholpon_ata_catchment.gpkg")

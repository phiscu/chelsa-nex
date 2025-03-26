import pandas as pd
from tqdm import tqdm
import configparser
import glob
import xarray as xr
import geopandas as gpd
import os
import pyproj
import pyproj
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
import xagg as xa

# Silence xagg prints
xa.set_options(silent=True)


def chelsa_w5e5_agg(shapefile_path, config_path="chelsa_config.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)

    base_dir = config["paths"]["base_dir"]
    target_subdir = config.get("output", "target_dir", fallback="aggregates")
    output_dir = os.path.join(base_dir, target_subdir)
    os.makedirs(output_dir, exist_ok=True)

    variables = [v.strip() for v in config["variables"]["vars"].split(",")]
    convert_temp = config.getboolean("conversion", "convert_temperatures")
    fill_value = 65535

    conversions = {
        "pr": {"factor": 86400, "offset": 0},
        "tas": {"factor": 1, "offset": -273.15 if convert_temp else 0},
        "tasmax": {"factor": 1, "offset": -273.15 if convert_temp else 0},
        "tasmin": {"factor": 1, "offset": -273.15 if convert_temp else 0},
        "rsds": {"factor": 1, "offset": 0},
    }

    polygon = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    poly_name = os.path.basename(shapefile_path).split('.')[0]

    for var in variables:
        print(f"\n ** Processing variable: {var}")
        files = sorted(glob.glob(f"{base_dir}/{var}/*.nc"))
        if not files:
            print(f"** No files found for {var}, skipping.")
            continue

        results = []
        previous_grid = None
        weightmap = None

        for file in tqdm(files, desc=f"→ {var}"):
            ds = xr.open_dataset(file)

            ds[var] = ds[var] * conversions[var]["factor"] + conversions[var]["offset"]
            ds[var] = ds[var].where(ds[var] != fill_value)

            # Extract current lat/lon grid
            current_grid = (tuple(ds["lat"].values), tuple(ds["lon"].values))

            # Only recalculate weights if grid changes
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
chelsa_w5e5_agg("/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/kyzylsuu/shp/kyzylsuu.shp")
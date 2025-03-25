import pandas as pd
from tqdm import tqdm
import configparser
import glob
import xarray as xr
import geopandas as gpd
import os
import pyproj
# To avoid error from regionmask on first import:
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
import regionmask


def chelsa_w5e5_agg(shapefile_path, config_path="chelsa_config.ini"):
    # Load config
    config = configparser.ConfigParser()
    config.read(config_path)

    base_dir = config["paths"]["base_dir"]
    target_subdir = config.get("output", "target_dir", fallback="aggregates")
    output_dir = os.path.join(base_dir, target_subdir)
    os.makedirs(output_dir, exist_ok=True)

    variables = [v.strip() for v in config["variables"]["vars"].split(",")]
    convert_temp = config.getboolean("conversion", "convert_temperatures")
    fill_value = 65535

    # Define conversion rules
    conversions = {
        "pr": {"factor": 86400, "offset": 0},
        "tas": {"factor": 1, "offset": -273.15 if convert_temp else 0},
        "tasmax": {"factor": 1, "offset": -273.15 if convert_temp else 0},
        "tasmin": {"factor": 1, "offset": -273.15 if convert_temp else 0},
        "rsds": {"factor": 1, "offset": 0},
    }

    # Load polygon
    polygon = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    poly_name = os.path.basename(shapefile_path).split('.')[0]

    for var in variables:
        print(f"\n ** Processing variable: {var}")
        files = sorted(glob.glob(f"{base_dir}/{var}/*.nc"))

        if not files:
            print(f"** No files found for {var}, skipping.")
            continue

        # Create mask once
        first_file = xr.open_dataset(files[0])
        mask = regionmask.Regions([polygon.geometry[0]])
        frac_mask = mask.mask_3D_frac_approx(first_file[var].isel(time=0))
        valid_mask = frac_mask.where(frac_mask > 0)
        first_file.close()

        results = []

        for file in tqdm(files, desc=f"→ {var}"):
            ds = xr.open_dataset(file)
            data = ds[var]

            # Apply conversion
            factor = conversions[var]["factor"]
            offset = conversions[var]["offset"]
            data = data * factor + offset

            # Mask fill values
            data = data.where(data != fill_value)

            # Weighted average
            weighted_sum = (data * valid_mask).sum(dim=("lat", "lon"), skipna=True)
            total_weight = valid_mask.sum(dim=("lat", "lon"), skipna=True)
            weighted_mean = weighted_sum / total_weight

            # Format output
            df = weighted_mean.to_dataframe(name=var).reset_index()
            df = df[["time", var]]
            df = round(df, 4)
            results.append(df)
            ds.close()

        # Combine and save
        final_df = pd.concat(results)
        output_file = os.path.join(output_dir, f"{poly_name}_{var}.csv")
        final_df.to_csv(output_file, index=False)
        print(f"** Saved: {output_file}")

    print("\n** Done.")


# Example usage
chelsa_w5e5_agg("/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/kyzylsuu/shp/kyzylsuu.shp")

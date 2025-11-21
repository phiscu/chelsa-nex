import configparser
import argparse
import os
import glob
import time
import sys
import geopandas as gpd
from matilda_functions import ClimateScenarios
from chelsa_preprocessing import chelsa_w5e5_agg
from era5l_preprocessing import ERA5LPolygonDownloader
import matplotlib
matplotlib.use('Agg')


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process geopackages for one batch.")
    parser.add_argument("--batch_dir", help="Path to the directory containing geopackages for one batch.")
    args = parser.parse_args()

    # Base config path
    base_config = "config.ini"

    # Load base settings
    config = configparser.ConfigParser()
    config.read(base_config)
    settings = config["settings"]

    # Use the parsed batch path if provided, otherwise use the one from the config
    try:
        batch_input = args.batch_dir if args.batch_dir else settings["geopackage_dir"]
    except KeyError:
        print("Error: No batch_dir argument provided and 'geopackage_dir' missing in config.ini.")
        sys.exit(1)

    # Detect whether input is a file or directory
    if os.path.isfile(batch_input):
        geopackages = [batch_input]
    elif os.path.isdir(batch_input):
        geopackages = sorted(glob.glob(os.path.join(batch_input, "*.gpkg")))
    else:
        print(f"Error: Path does not exist: {batch_input}")
        sys.exit(1)

    reanalysis = settings["reanalysis"]
    era5l_dir = settings["era5l_dir"]
    chelsa_dir = settings["chelsa_dir"]
    cmip_dir = settings["output_dir"]
    gee_project = settings['gee_project']
    download = settings.getboolean("download")
    show = settings.getboolean("show")
    load_backup = settings.getboolean("load_backup")
    processes = settings.getint("processes")
    starty = settings.getint("start_year")
    endy = settings.getint("end_year")
    force_preprocessing = settings.getboolean("force_preprocessing")
    plots = settings.getboolean("plots")
    convert_temp = config.getboolean("reanalysis", "convert_temperatures")

    os.makedirs(cmip_dir, exist_ok=True)
    print(f"Output directory for CMIP6: {cmip_dir}")

    print(f"Found {len(geopackages)} input file(s) in '{batch_input}'")

    total_start = time.time()

    # Loop over each geopackage
    for gpkg in geopackages:
        print(f"\n=== Processing {gpkg} ===")

        gdf = gpd.read_file(gpkg)
        if 'RGIId' not in gdf.columns:
            print(f"Error: GeoPackage {gpkg} must contain an 'RGIId' column.")
            continue

        for _, row in gdf.iterrows():
            poly_id = row['RGIId']
            poly_name = f"RGI-{poly_id}"
            print(f"\n--- Processing polygon {poly_name} ---")

            # Write temporary GPKG with single polygon
            temp_gpkg = os.path.join("/tmp", f"{poly_name}.gpkg")
            row_gdf = gpd.GeoDataFrame([row], crs=gdf.crs)
            row_gdf.to_file(temp_gpkg, driver="GPKG")

            if reanalysis == "era5l":
                reanalysis_dir = era5l_dir
            elif reanalysis == "chelsa":
                reanalysis_dir = chelsa_dir
            else:
                print("Reanalysis type not recognized. Choose either 'era5l' or 'chelsa'.")
                continue

            # Run CMIP6 bias correction workflow
            instance = ClimateScenarios(
                output=cmip_dir,
                reanalysis=reanalysis,
                reanalysis_dir=reanalysis_dir,
                polygon_path=temp_gpkg,
                gee_project=gee_project,
                download=download,
                load_backup=load_backup,
                show=show,
                starty=starty,
                endy=endy,
                processes=processes,
                plots=plots
            )
            instance.complete_workflow()


    total_end = time.time()
    print(f"\nTotal processing time: {total_end - total_start:.2f} seconds")

if __name__ == '__main__':
    main()

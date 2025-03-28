import configparser
import os
import glob
import time
from matilda_functions import ClimateScenarios
from chelsa_preprocessing import chelsa_w5e5_agg
import matplotlib
matplotlib.use('Agg')

def main():

    # Base config path
    base_config = "config.ini"

    # Load base settings
    config = configparser.ConfigParser()
    config.read(base_config)
    settings = config["settings"]

    chelsa_dir = settings["chelsa_dir"]
    cmip_dir = settings["output_dir"]
    geopackage_dir = settings["geopackage_dir"]
    gee_project = settings['gee_project']
    download = settings.getboolean("download")
    show = settings.getboolean("show")
    load_backup = settings.getboolean("load_backup")
    processes = settings.getint("processes")
    starty = settings.getint("start_year")
    endy = settings.getint("end_year")
    force_chelsa = settings.getboolean("force_chelsa")
    plots = settings.getboolean("plots")

    # Folder with geopackages
    geopackages = sorted(glob.glob(os.path.join(geopackage_dir, "*.gpkg")))
    print(f"Found {len(geopackages)} GeoPackages")

    total_start = time.time()

    # Loop over each geopackage
    for gpkg in geopackages:
        print(f"\n=== Processing {gpkg} ===")

        # Run CHELSA aggregation
        chelsa_w5e5_agg(gpkg, config_path="chelsa_config.ini", force=force_chelsa)

        # Update polygon path in ClimateScenarios config
        instance = ClimateScenarios(
            output=cmip_dir,
            chelsa_dir=chelsa_dir,
            polygon_path=gpkg,
            gee_project=gee_project,
            download=download,
            load_backup=load_backup,
            show=show,
            starty=starty,
            endy=endy,
            processes=processes,
            plots=plots
        )

        # Run CMIP6 workflow
        instance.complete_workflow()

    total_end = time.time()
    print(f"\nTotal processing time: {total_end - total_start:.2f} seconds")

if __name__ == '__main__':
    main()

import os
import shutil

def collect_dat_files(base_dir, model_names, output_dir):
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all subdirectories in the base directory
    for site in os.listdir(base_dir):
        site_path = os.path.join(base_dir, site)
        dat_folder = os.path.join(site_path, "dat_files")

        if not os.path.isdir(dat_folder):
            continue  # skip if not a valid site dir

        # Create corresponding site subfolder in output
        output_site_dir = os.path.join(output_dir, site)
        os.makedirs(output_site_dir, exist_ok=True)

        for model in model_names:
            for ssp in ["SSP2", "SSP5"]:
                dat_file = f"{model}_{ssp}.dat"
                src_path = os.path.join(dat_folder, dat_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(output_site_dir, dat_file))

if __name__ == "__main__":
    # Set paths
    base_directory = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/climate/cmip6"  # replace with your actual path
    output_directory = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/climate/glogem"    # where to save the filtered files

    # List your models (without SSP or file extension)
    models_to_include = [
	    "GFDL-ESM4",
	    "UKESM1-0-LL",
	    "MPI-ESM1-2-HR",
	    "IPSL-CM6A-LR",
	    "MRI-ESM2-0"
	]


    collect_dat_files(base_directory, models_to_include, output_directory)


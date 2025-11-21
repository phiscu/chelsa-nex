import os
import ee
import requests
import datetime
import time
import concurrent.futures
import geopandas as gpd
import pandas as pd
import glob
from tqdm import tqdm
import geemap
import pyproj

# To avoid PROJ error:
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()


class ERA5LPolygonDownloader:

    def __init__(self, polygon_path, gee_project, era5l_dir, convert_temp, force=False):
        self.polygon_path = polygon_path
        self.poly_name = os.path.splitext(os.path.basename(polygon_path))[0]
        self.gee_project = gee_project
        self.era5l_dir = era5l_dir
        self.convert_temp = convert_temp
        self.force = force

    def initialize_target(self):
        try:
            ee.Initialize(project=self.gee_project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=self.gee_project)
        shape = gpd.read_file(self.polygon_path)
        self.polygon_ee = geemap.geopandas_to_ee(shape)

    def setProperty(self, image):

        dict = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=self.polygon_ee,
            scale=500                   # Scale to avoid problems with small polygons
        )
        return image.set(dict)

    def download_era5l(self):
        output_file = os.path.join(self.era5l_dir, f"era5l_{self.poly_name}.csv")

        # Skip if already exists and not forced
        if os.path.exists(output_file) and not self.force:
            print(f"ERA5-Land output for '{self.poly_name}' already exists. Skipping.")
            return

        self.initialize_target()

        collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_RAW') \
            .select('temperature_2m', 'total_precipitation_sum')

        withMean = collection.map(self.setProperty)

        # Download and write to dataframe
        df = pd.DataFrame()
        print("Get timestamps...")
        df['ts'] = withMean.aggregate_array('system:time_start').getInfo()
        df['dt'] = df['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))
        print("Get temperature values...")
        df['temp'] = withMean.aggregate_array('temperature_2m').getInfo()
        if self.convert_temp:
            df['temp_c'] = df['temp'] - 273.15
        print("Get precipitation values...")
        df['prec'] = withMean.aggregate_array('total_precipitation_sum').getInfo()
        df['prec'] = df['prec'] * 1000

        df.to_csv(output_file, header=True, index=False)
        print(f"** Saved: {output_file}")


class ERA5LParallelDownloader:
    def __init__(self, polygon_input, out_dir, start_year, end_year, processes=8, max_retries=3):
        self.polygon_input = polygon_input
        self.out_dir = out_dir
        self.start_year = start_year
        self.end_year = end_year
        self.processes = processes
        self.max_retries = max_retries
        self.expected_years = list(range(start_year, end_year + 1))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.gdf = self._load_polygons()
        print(f"Loaded {len(self.gdf)} polygons.")
        ee.Initialize()

    def _load_polygons(self):
        if os.path.isdir(self.polygon_input):
            files = [os.path.join(self.polygon_input, f)
                     for f in os.listdir(self.polygon_input) if f.endswith((".shp", ".gpkg"))]
            gdfs = []
            for f in files:
                gdf = gpd.read_file(f)
                gdf['poly_name'] = os.path.splitext(os.path.basename(f))[0]
                gdfs.append(gdf)
            merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        else:
            gdf = gpd.read_file(self.polygon_input)
            if 'RGIId' not in gdf.columns:
                raise ValueError("Input file must contain an 'RGIId' column.")
            gdf = gdf.reset_index(drop=True)
            gdf['poly_name'] = gdf['RGIId'].apply(lambda x: f"RGI-{x}")
            merged = gdf

        merged['poly_id'] = merged.index.astype(str)
        return merged

    def _build_url_and_download(self, poly_idx, year):
        row = self.gdf.iloc[poly_idx]
        poly_name = row['poly_name']
        poly_dir = os.path.join(self.out_dir, poly_name)
        os.makedirs(poly_dir, exist_ok=True)
        output_file = os.path.join(poly_dir, f'{year}.csv')

        if os.path.exists(output_file):
            return f"Skipped (exists): {poly_name} {year}"

        geom = geemap.geopandas_to_ee(gpd.GeoDataFrame([row], crs=self.gdf.crs))

        start = f"{year}-01-01"
        end = f"{year+1}-01-01"

        collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_RAW") \
            .filterDate(start, end) \
            .select(['temperature_2m', 'total_precipitation_sum'])

        def build_feature(img):
            date = img.date().format('YYYY-MM-dd')
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom.geometry(),
                scale=500
            )
            return ee.Feature(None, stats.set('date', date))

        try:
            days = ee.List(collection.toList(collection.size()))
            fc = ee.FeatureCollection(days.map(lambda i: build_feature(ee.Image(i))))
        except Exception as e:
            return f"Failed to build FeatureCollection: {poly_name} {year} ({e})"

        for attempt in range(1, self.max_retries + 1):
            try:
                url = fc.getDownloadURL(filetype='CSV')
                r = requests.get(url)
                if r.status_code == 200:
                    with open(output_file, 'w') as f:
                        f.write(r.text)
                    return
                else:
                    r.raise_for_status()
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed after retries: {poly_name} {year} ({e})"
                time.sleep(2 ** attempt)

    def _merge_csvs(self):
        log = []
        for _, row in self.gdf.iterrows():
            poly_name = row['poly_name']
            poly_dir = os.path.join(self.out_dir, poly_name)
            merged_path = os.path.join(self.out_dir, f"era5l_{poly_name}.csv")

            if not os.path.exists(poly_dir):
                log.append({'polygon': poly_name, 'status': 'missing directory'})
                continue

            year_files = sorted(glob.glob(os.path.join(poly_dir, "*.csv")))
            found_years = [int(os.path.basename(f).replace(".csv", "")) for f in year_files if os.path.basename(f).replace(".csv", "").isdigit()]

            missing_years = sorted(set(self.expected_years) - set(found_years))

            if missing_years:
                print(f"Retrying missing years for {poly_name}: {missing_years}")
                poly_idx = self.gdf[self.gdf['poly_name'] == poly_name].index[0]
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.processes) as executor:
                    futures = [executor.submit(self._build_url_and_download, poly_idx, y) for y in missing_years]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            print(result)

                # Recheck after retry
                year_files = sorted(glob.glob(os.path.join(poly_dir, "*.csv")))
                found_years = [int(os.path.basename(f).replace(".csv", "")) for f in year_files if os.path.basename(f).replace(".csv", "").isdigit()]
                missing_years = sorted(set(self.expected_years) - set(found_years))

            if missing_years:
                log.append({'polygon': poly_name, 'status': f"missing years: {missing_years}"})
                continue

            dfs = []
            for yf in sorted(year_files):
                df = pd.read_csv(yf)
                df.rename(columns={
                    'temperature_2m': 'temp',
                    'total_precipitation_sum': 'prec'
                }, inplace=True)
                if 'prec' in df.columns:
                    df['prec'] = df['prec'] * 1000
                df.drop(columns=["system:index", ".geo"], errors="ignore", inplace=True)
                dfs.append(df)

            df_merged = pd.concat(dfs)
            df_merged.sort_values('date', inplace=True)
            df_merged.to_csv(merged_path, index=False)
            print(f"Merged and saved: {merged_path}")
            log.append({'polygon': poly_name, 'status': 'complete'})

        # Construct log filename using basename of input file (without extension)
        base = os.path.splitext(os.path.basename(self.polygon_input))[0]
        log_path = os.path.join(self.out_dir, f"era5l_download_log_{base}.csv")

        log_df = pd.DataFrame(log)
        log_df.to_csv(log_path, index=False)
        print(f"Wrote status log to {log_path}.")

    def run(self):
        tasks = [(i, y) for i in range(len(self.gdf)) for y in self.expected_years]
        print(f"Launching {len(tasks)} download jobs with {self.processes} processes...")

        with tqdm(total=len(tasks), desc="Downloading ERA5-Land CSVs") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.processes) as executor:
                futures = [executor.submit(self._build_url_and_download, i, y) for i, y in tasks]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        tqdm.write(result)
                    pbar.update(1)

        print("Merging annual CSVs into one file per polygon...")
        self._merge_csvs()


##
# downloader = ERA5LParallelDownloader(
#     # polygon_input="/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/kyzylsuu/gpkg_test",    # or a folder of shapefiles
#     # polygon_input="/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/kyzylsuu/gpkg_glaciers",
#     polygon_input="/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/glaciers/issyk_kul/multiple_test.gpkg",
#     out_dir="/home/phillip/Seafile/EBA-CA/Repositories/chelsa-nex/era5l_test",
#     start_year=1950,
#     end_year=1955,
#     processes=10,
#     max_retries=3
# )
# downloader.run()


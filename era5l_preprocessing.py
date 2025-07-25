import pandas as pd
import datetime
import ee
import geemap
import geopandas as gpd
import os
import pyproj

# To avoid PROJ error:
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()


class ERA5LPolygonDownloader:

    def __init__(self, polygon_path, gee_project, era5l_dir): #starty, endy,
        self.polygon_path = polygon_path
        self.poly_name = os.path.splitext(os.path.basename(polygon_path))[0]
        self.gee_project = gee_project
        # self.starty = starty
        # self.endy = endy
        self.era5l_dir = era5l_dir

    def initialize_target(self):
        try:
            ee.Initialize(project=self.gee_project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=self.gee_project)
        shape = gpd.read_file(self.polygon_path)
        self.polygon_ee = geemap.geopandas_to_ee(shape)

    def setProperty(self, image):
        dict = image.reduceRegion(ee.Reducer.mean(), self.polygon_ee)
        return image.set(dict)

    def download_era5l(self):
        self.initialize_target()

        # Create FeatureCollection from polygon
        collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_RAW') \
            .select('temperature_2m', 'total_precipitation_sum') \
            # .filterDate(self.starty, self.endy)
        withMean = collection.map(self.setProperty)

        # Download and write to dataframe
        df = pd.DataFrame()
        print("Get timestamps...")
        df['ts'] = withMean.aggregate_array('system:time_start').getInfo()
        df['dt'] = df['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))
        print("Get temperature values...")
        df['temp'] = withMean.aggregate_array('temperature_2m').getInfo()
        df['temp_c'] = df['temp'] - 273.15
        print("Get precipitation values...")
        df['prec'] = withMean.aggregate_array('total_precipitation_sum').getInfo()
        df['prec'] = df['prec'] * 1000

        output_file = os.path.join(self.era5l_dir, f"era5l_{self.poly_name}.csv")
        df.to_csv(output_file, header=True, index=False)
        print(f"** Saved: {output_file}")

# Test the class
gee_project = 'matilda-edu'
polygon_path = '/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/kyzylsuu/gpkg_test/RGI60-13.07926.gpkg'
era5l_dir = '/home/phillip/Seafile/EBA-CA/Repositories/chelsa-nex/era5l_test'

processor = ERA5LPolygonDownloader(polygon_path, gee_project, era5l_dir)
processor.download_era5l()


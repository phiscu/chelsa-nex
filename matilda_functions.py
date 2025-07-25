import os
import pyproj
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
import requests
import concurrent.futures
import seaborn as sns
import probscale
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import geopandas as gpd
import ee
import geemap
import pickle
import numpy as np
from retry import retry
from tqdm import tqdm
from bias_correction import BiasCorrection
from matplotlib.legend import Legend

# warnings.filterwarnings("ignore")


class CMIPDownloader:
    """Class to download spatially averaged CMIP6 data for a given period, variable, and spatial subset."""

    def __init__(self, var, starty, endy, shape, processes=10, dir='./'):
        self.var = var
        self.starty = starty
        self.endy = endy
        self.shape = shape
        self.processes = processes
        self.directory = dir

        # create the download directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def get_area(self):
        """Load a polygon from file and calculate area in square meters.
        Stores the area as self.area_m2.
        """
        # Compute area from EE geometry (client-side)
        try:
            self.area_m2 = self.shape.geometry().area().getInfo()
        except Exception as e:
            print("Could not retrieve area from EE geometry:", e)
            self.area_m2 = None
        print(f'Polygon area: {round(self.area_m2, 2)} m²')

    def download(self):
        """Runs a subset routine for CMIP6 data on GEE servers to create ee.FeatureCollections for all years in
        the requested period. Downloads individual years in parallel processes to decrease the download time."""

        print('Initiating download request for NEX-GDDP-CMIP6 data from ' +
              str(self.starty) + ' to ' + str(self.endy) + '.')

        self.get_area()

        use_fallback = self.area_m2 < 2e6  # client-side check for small polygons

        print('Scaling to 500m for small polygons.' if use_fallback else 'Scaling to native resolution for large polygons.')

        def getRequests(starty, endy):
            """Generates a list of years to be downloaded. [Client side]"""

            return [i for i in range(starty, endy + 1)]

        @retry(tries=10, delay=1, backoff=2)
        def getResult(index, year):
            """Handle the HTTP requests to download one year of CMIP6 data. [Server side]"""

            start = str(year) + '-01-01'
            end = str(year + 1) + '-01-01'
            startDate = ee.Date(start)
            endDate = ee.Date(end)
            n = endDate.difference(startDate, 'day').subtract(1)

            def getImageCollection(var):
                """Create and image collection of CMIP6 data for the requested variable, period, and region.
                [Server side]"""

                collection = ee.ImageCollection('NASA/GDDP-CMIP6') \
                    .select(var) \
                    .filterDate(startDate, endDate)
                    # .filterBounds(self.shape)
                return collection

            def renameBandName(b):
                """Edit variable names for better readability. [Server side]"""

                split = ee.String(b).split('_')
                return ee.String(split.splice(split.length().subtract(2), 1).join("_"))

            def buildFeature(i):
                """Extract daily data with strategy based on polygon area. [Server side]"""

                t1 = startDate.advance(i, 'day')
                t2 = t1.advance(1, 'day')
                dailyColl = collection.filterDate(t1, t2)
                dailyImg = dailyColl.toBands()

                # Rename bands
                bands = dailyImg.bandNames()
                renamed = bands.map(renameBandName)
                image = dailyImg.rename(renamed)

                if use_fallback:
                    result_dict = image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=self.shape,
                        scale=500
                    )
                else:
                    result_dict = image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=self.shape,
                        scale=500             # in few cases native resolution fails nonetheless, so 500 for all...
                    )

                result_dict = result_dict.combine(
                    ee.Dictionary({
                        'system:time_start': t1.millis(),
                        'isodate': t1.format('YYYY-MM-dd')})
                )

                return ee.Feature(None, result_dict)

            # Create features for all days in the respective year. [Server side]
            collection = getImageCollection(self.var)
            year_feature = ee.FeatureCollection(ee.List.sequence(0, n).map(buildFeature))

            # Create a download URL for a CSV containing the feature collection. [Server side]
            url = year_feature.getDownloadURL()

            # Handle downloading the actual csv for one year. [Client side]
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                r.raise_for_status()
            filename = os.path.join(self.directory, 'cmip6_' + self.var + '_' + str(year) + '.csv')
            with open(filename, 'w') as f:
                f.write(r.text)

            return index

        # Create a list of years to be downloaded. [Client side]
        items = getRequests(self.starty, self.endy)

        # Launch download requests in parallel processes and display a status bar. [Client side]
        with tqdm(total=len(items), desc="Downloading CMIP6 data for variable '" + self.var + "'") as pbar:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.processes) as executor:
                for i, year in enumerate(items):
                    results.append(executor.submit(getResult, i, year))
                for future in concurrent.futures.as_completed(results):
                    index = future.result()
                    pbar.update(1)

        print("All downloads complete.")


class CMIPProcessor:
    """Class to read and pre-process CSV files downloaded by the CMIPDownloader class."""

    def __init__(self, var, file_dir='.', start=1979, end=2100):
        self.file_dir = file_dir
        self.var = var
        self.start = start
        self.end = end
        self.df_hist = self.append_df(self.var, self.start, self.end, self.file_dir, hist=True)
        self.df_ssp = self.append_df(self.var, self.start, self.end, self.file_dir, hist=False)
        self.ssp2_common, self.ssp5_common, self.hist_common, \
            self.common_models, self.dropped_models = self.process_dataframes()
        self.ssp2, self.ssp5 = self.get_results()

    def read_cmip(self, filename):
        """Reads CMIP6 CSV files and drops redundant columns."""

        df = pd.read_csv(filename, index_col='isodate', parse_dates=['isodate'])
        df = df.drop(['system:index', '.geo', 'system:time_start'], axis=1)
        return df

    def append_df(self, var, start, end, file_dir='.', hist=True):
        """Reads CMIP6 CSV files of individual years and concatenates them into dataframes for the full downloaded
        period. Historical and scenario datasets are treated separately. Converts precipitation unit to mm."""

        df_list = []
        if hist:
            starty = start
            endy = 2014
        else:
            starty = 2015
            endy = end
        for i in range(starty, endy + 1):
            filename = file_dir + 'cmip6_' + var + '_' + str(i) + '.csv'
            df_list.append(self.read_cmip(filename))
        if hist:
            hist_df = pd.concat(df_list)
            if var == 'pr':
                hist_df = hist_df * 86400  # from kg/(m^2*s) to mm/day
            return hist_df
        else:
            ssp_df = pd.concat(df_list)
            if var == 'pr':
                ssp_df = ssp_df * 86400  # from kg/(m^2*s) to mm/day
            return ssp_df

    def process_dataframes(self):
        """Separates the two scenarios and drops models not available for both scenarios and the historical period."""

        ssp2 = self.df_ssp.loc[:, self.df_ssp.columns.str.startswith('ssp245')]
        ssp5 = self.df_ssp.loc[:, self.df_ssp.columns.str.startswith('ssp585')]
        hist = self.df_hist.loc[:, self.df_hist.columns.str.startswith('historical')]

        ssp2.columns = ssp2.columns.str.lstrip('ssp245_').str.rstrip('_' + self.var)
        ssp5.columns = ssp5.columns.str.lstrip('ssp585_').str.rstrip('_' + self.var)
        hist.columns = hist.columns.str.lstrip('historical_').str.rstrip('_' + self.var)

        # Get all the models the three datasets have in common
        common_models = set(ssp2.columns).intersection(ssp5.columns).intersection(hist.columns)

        # Get the model names that contain NaN values
        nan_models_list = [df.columns[df.isna().any()].tolist() for df in [ssp2, ssp5, hist]]
        # flatten the list
        nan_models = [col for sublist in nan_models_list for col in sublist]
        # remove duplicates
        nan_models = list(set(nan_models))

        # Remove models with NaN values from the list of common models
        common_models = [x for x in common_models if x not in nan_models]

        ssp2_common = ssp2.loc[:, common_models]
        ssp5_common = ssp5.loc[:, common_models]
        hist_common = hist.loc[:, common_models]

        dropped_models = list(set([mod for mod in ssp2.columns if mod not in common_models] +
                                  [mod for mod in ssp5.columns if mod not in common_models] +
                                  [mod for mod in hist.columns if mod not in common_models]))

        return ssp2_common, ssp5_common, hist_common, common_models, dropped_models

    def get_results(self):
        """Concatenates historical and scenario data to combined dataframes of the full downloaded period.
        Arranges the models in alphabetical order."""

        ssp2_full = pd.concat([self.hist_common, self.ssp2_common])
        ssp2_full.index.names = ['TIMESTAMP']
        ssp5_full = pd.concat([self.hist_common, self.ssp5_common])
        ssp5_full.index.names = ['TIMESTAMP']

        ssp2_full = ssp2_full.reindex(sorted(ssp2_full.columns), axis=1)
        ssp5_full = ssp5_full.reindex(sorted(ssp5_full.columns), axis=1)

        return ssp2_full, ssp5_full


def adjust_bias(predictand, predictor, datasource='era5',
                train_start='1979-01-01', train_end='2022-12-31',
                method='normal_mapping'):
    """
    Adjusts for the bias between target data and the historic CMIP6 model runs.
    Optionally replaces training period with CHELSA observations (1979–2016) for consistent output.
    """

    # Read predictor data
    if datasource == 'era5':
        predictor = read_era5l(predictor)
        var = 'temp' if predictand.mean().mean() > 100 else 'prec'
    elif datasource == 'chelsa':
        var = 'tas' if predictand.mean().mean() > 100 else 'pr'
    else:
        var = 'temp' if predictand.mean().mean() > 100 else 'prec'

    # Initialize correction periods
    correction_periods = [
        {'correction_range': ('1979-01-01', '2010-12-31'), 'extraction_range': ('1979-01-01', '1990-12-31')},
    ]
    for decade_start in range(1991, 2090, 10):
        correction_periods.append({
            'correction_range': (f"{decade_start - 10}-01-01", f"{decade_start + 19}-12-31"),
            'extraction_range': (f"{decade_start}-01-01", f"{decade_start + 9}-12-31")
        })
    correction_periods.append({
        'correction_range': ('2081-01-01', '2100-12-31'),
        'extraction_range': ('2091-01-01', '2100-12-31')
    })

    # Initialize empty DataFrame for corrected data
    corrected_data = pd.DataFrame()

    # Shared training period
    training_period = slice(train_start, train_end)

    for period in correction_periods:
        correction_start, correction_end = period['correction_range']
        extraction_start, extraction_end = period['extraction_range']

        correction_slice = slice(correction_start, correction_end)
        extraction_slice = slice(extraction_start, extraction_end)

        data_corr = pd.DataFrame()

        for col in predictand.columns:
            x_train = predictand[col][training_period].squeeze()
            y_train = predictor[training_period][var].squeeze()
            x_predict = predictand[col][correction_slice].squeeze()

            bc_corr = BiasCorrection(y_train, x_train, x_predict)
            corrected_col = pd.DataFrame(bc_corr.correct(method=method))

            data_corr[col] = corrected_col.loc[extraction_slice]

        # Clamp negative precipitation to 0
        if var in ['prec', 'pr']:
            data_corr[data_corr < 0] = 0

        corrected_data = pd.concat([corrected_data, data_corr])

    # Replace 1979–2016 with CHELSA if applicable
    if datasource == 'chelsa':
        # Extract raw and adjusted CMIP6 for training period (for later comparison)
        raw_train = predictand.loc['1979-01-01':'2016-12-31']
        corrected_train = corrected_data.loc['1979-01-01':'2016-12-31']

        # Extract CHELSA data and replicate across columns
        predictor_hist = predictor.loc['1979-01-01':'2016-12-31', var].to_frame()
        predictor_hist = pd.DataFrame(
            np.tile(predictor_hist.values, (1, len(predictand.columns))),
            columns=predictand.columns,
            index=predictor_hist.index
        )
        predictor_hist.index.name = 'TIMESTAMP'

        # Combine with corrected future data
        corrected_future = corrected_data.loc['2017-01-01':]
        corrected_data = pd.concat([predictor_hist, corrected_future])
        corrected_data.sort_index(inplace=True)
        corrected_data.index.name = 'TIMESTAMP'

        return corrected_data, raw_train, corrected_train

    else:
        corrected_data.sort_index(inplace=True)
        corrected_data.index.name = 'TIMESTAMP'
        return corrected_data



class CMIP6PolygonProcessor:
    """
    Fork of the CMIP6DataProcessor to work with a general polygon and CHELSA-W5E5 data instead of weather stations.
    """

    def __init__(self, polygon_path, gee_project, starty, endy, cmip_dir, reanalysis_dir, processes):
        self.polygon_path = polygon_path
        self.poly_name = os.path.splitext(os.path.basename(polygon_path))[0]
        self.gee_project = gee_project
        self.starty = starty
        self.endy = endy
        self.cmip_dir = cmip_dir
        self.reanalysis_dir = reanalysis_dir
        self.processes = processes

    def initialize_target(self):
        try:
            ee.Initialize(project=self.gee_project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=self.gee_project)
        shape = gpd.read_file(self.polygon_path)
        self.polygon_ee = geemap.geopandas_to_ee(shape)

    def download_cmip6_data(self):
        self.initialize_target()
        downloader_t = CMIPDownloader('tas', self.starty, self.endy, self.polygon_ee, self.processes, self.cmip_dir)
        downloader_t.download()
        downloader_p = CMIPDownloader('pr', self.starty, self.endy, self.polygon_ee, self.processes, self.cmip_dir)
        downloader_p.download()

    def process_cmip6_data(self):
        processor_t = CMIPProcessor('tas', self.cmip_dir,  self.starty, self.endy)
        self.ssp2_tas_raw, self.ssp5_tas_raw = processor_t.get_results()
        processor_p = CMIPProcessor('pr', self.cmip_dir, self.starty, self.endy)
        self.ssp2_pr_raw, self.ssp5_pr_raw = processor_p.get_results()

    def bias_adjustment(self):
        self.initialize_target()
        self.process_cmip6_data()
        print('Running bias adjustment routine...')

        # Load CHELSA data
        tas_chelsa = pd.read_csv(f'{self.reanalysis_dir}{self.poly_name}_tas.csv', index_col='time', parse_dates=True)
        pr_chelsa = pd.read_csv(f'{self.reanalysis_dir}{self.poly_name}_pr.csv', index_col='time', parse_dates=True)

        train_start_temp = str(tas_chelsa.first_valid_index())
        train_end_temp = str(tas_chelsa.last_valid_index())
        train_start_prec = str(pr_chelsa.first_valid_index())
        train_end_prec = str(pr_chelsa.last_valid_index())

        # Temperature bias adjustment
        self.ssp2_tas, raw2_tas_train, adj2_tas_train = adjust_bias(
            self.ssp2_tas_raw, tas_chelsa, datasource='chelsa',
            train_start=train_start_temp, train_end=train_end_temp
        )
        self.ssp5_tas, raw5_tas_train, adj5_tas_train = adjust_bias(
            self.ssp5_tas_raw, tas_chelsa, datasource='chelsa',
            train_start=train_start_temp, train_end=train_end_temp
        )

        # Precipitation bias adjustment
        self.ssp2_pr, raw2_pr_train, adj2_pr_train = adjust_bias(
            self.ssp2_pr_raw, pr_chelsa, datasource='chelsa',
            train_start=train_start_prec, train_end=train_end_prec
        )
        self.ssp5_pr, raw5_pr_train, adj5_pr_train = adjust_bias(
            self.ssp5_pr_raw, pr_chelsa, datasource='chelsa',
            train_start=train_start_prec, train_end=train_end_prec
        )

        # Main dictionaries for raw and adjusted CMIP6 data
        self.ssp_tas_dict = {
            'SSP2_raw': self.ssp2_tas_raw,
            'SSP2_adjusted': self.ssp2_tas,
            'SSP5_raw': self.ssp5_tas_raw,
            'SSP5_adjusted': self.ssp5_tas
        }

        self.ssp_pr_dict = {
            'SSP2_raw': self.ssp2_pr_raw,
            'SSP2_adjusted': self.ssp2_pr,
            'SSP5_raw': self.ssp5_pr_raw,
            'SSP5_adjusted': self.ssp5_pr
        }

        # Dedicated training evaluation dictionary
        self.training_dict = {
            'tas': {
                'SSP2_raw_train': raw2_tas_train,
                'SSP2_adjusted_train': adj2_tas_train,
                'SSP5_raw_train': raw5_tas_train,
                'SSP5_adjusted_train': adj5_tas_train,
                'reference': tas_chelsa.loc[train_start_temp:train_end_temp]
            },
            'pr': {
                'SSP2_raw_train': raw2_pr_train,
                'SSP2_adjusted_train': adj2_pr_train,
                'SSP5_raw_train': raw5_pr_train,
                'SSP5_adjusted_train': adj5_pr_train,
                'reference': pr_chelsa.loc[train_start_prec:train_end_prec]
            }
        }

        print('Done!')


def process_nested_dict(d, func, *args, **kwargs):
    for key, value in d.items():
        if isinstance(value, pd.DataFrame):
            d[key] = func(value, *args, **kwargs)
        elif isinstance(value, dict):
            process_nested_dict(value, func, *args, **kwargs)


def dict_filter(dictionary, filter_string):
    """Returns a dict with all elements of the input dict that contain a filter string in their keys."""
    return {key.split('_')[0]: value for key, value in dictionary.items() if filter_string in key}


class DataFilter:
    def __init__(self, df, zscore_threshold=3, resampling_rate=None, prec=False, jump_threshold=5):
        self.df = df
        self.zscore_threshold = zscore_threshold
        self.resampling_rate = resampling_rate
        self.prec = prec
        self.jump_threshold = jump_threshold
        self.filter_all()

    def check_outliers(self):
        """
        A function for filtering a pandas dataframe for columns with obvious outliers
        and dropping them based on a z-score threshold.

        Returns
        -------
        models : list
            A list of columns identified as having outliers.
        """
        # Resample if rate specified
        if self.resampling_rate is not None:
            if self.prec:
                self.df = self.df.resample(self.resampling_rate).sum()
            else:
                self.df = self.df.resample(self.resampling_rate).mean()

        # Calculate z-scores for each column
        z_scores = pd.DataFrame((self.df - self.df.mean()) / self.df.std())

        # Identify columns with at least one outlier (|z-score| > threshold)
        cols_with_outliers = z_scores.abs().apply(lambda x: any(x > self.zscore_threshold))
        self.outliers = list(self.df.columns[cols_with_outliers])

        # Return the list of columns with outliers
        return self.outliers

    def check_jumps(self):
        """
        A function for checking a pandas dataframe for columns with sudden jumps or drops
        and returning a list of the columns that have them.

        Returns
        -------
        jumps : list
            A list of columns identified as having sudden jumps or drops.
        """
        cols = self.df.columns
        jumps = []

        for col in cols:
            diff = self.df[col].diff()
            if (abs(diff) > self.jump_threshold).any():
                jumps.append(col)

        self.jumps = jumps
        return self.jumps

    def filter_all(self):
        """
        A function for filtering a dataframe for columns with obvious outliers
        or sudden jumps or drops in temperature, and returning a list of the
        columns that have been filtered using either or both methods.

        Returns
        -------
        filtered_models : list
            A list of columns identified as having outliers or sudden jumps/drops in temperature.
        """
        self.check_outliers()
        self.check_jumps()
        self.filtered_models = list(set(self.outliers) | set(self.jumps))
        return self.filtered_models


def loop_checks(ssp_dict, **kwargs):
    """
    Wrapper for class DataFilter to iterate over all scenarios.
    """
    outliers = []
    jumps = []
    both_checks = []
    for scenario in ssp_dict.keys():
        filter = DataFilter(ssp_dict[scenario], **kwargs)
        outliers.extend(set(filter.outliers))
        jumps.extend(set(filter.jumps))
        both_checks.extend(set(filter.filtered_models))

    return outliers, jumps, both_checks


def drop_model(col_names, dict_or_df):
    """
    Drop columns with given names from either a dictionary of dataframes
    or a single dataframe.
    Parameters
    ----------
    col_names : list of str
        The list of model names to drop.
    dict_or_df : dict of pandas.DataFrame or pandas.DataFrame
        If a dict of dataframes, all dataframes in the dict will be edited.
        If a single dataframe, only that dataframe will be edited.
    Returns
    -------
    dict_of_dfs : dict of pandas.DataFrame or pandas.DataFrame
        The updated dictionary of dataframes or dataframe with dropped columns.
    """
    if isinstance(dict_or_df, dict):
        # loop through the dictionary and edit each dataframe
        for key in dict_or_df.keys():
            if all(col_name in dict_or_df[key].columns for col_name in col_names):
                dict_or_df[key] = dict_or_df[key].drop(columns=col_names)
        return dict_or_df
    elif isinstance(dict_or_df, pd.DataFrame):
        # edit the single dataframe
        if all(col_name in dict_or_df.columns for col_name in col_names):
            return dict_or_df.drop(columns=col_names)
    else:
        raise TypeError('Input must be a dictionary or a dataframe')


def apply_filters(temp_dict, prec_dict, **kwargs):
    """
    Applies data filters to temperature and precipitation dictionaries.
    Parameters
    ----------
    temp_dict : dict
        Dictionary containing temperature data.
    prec_dict : dict
        Dictionary containing precipitation data.
    **kwargs
        Additional keyword arguments for loop_checks function.
    Returns
    -------
    tuple of pandas.DataFrame
        Tuple containing filtered temperature and precipitation data.
    """
    tas_raw = dict_filter(temp_dict, 'raw')
    outliers, jumps, both_checks = loop_checks(tas_raw, **kwargs)
    print('Applying data filters...')
    print('Models with temperature outliers: ' + str(outliers))
    print('Models with temperature jumps: ' + str(jumps))
    print('Models excluded: ' + str(both_checks))

    return drop_model(both_checks, temp_dict), drop_model(both_checks, prec_dict)


def pickle_to_dict(file_path):
    """
    Loads a dictionary from a pickle file at a specified file path.
    Parameters
    ----------
    file_path : str
        The path of the pickle file to load.
    Returns
    -------
    dict
        The dictionary loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        dic = pickle.load(f)
    return dic


def dict_to_pickle(dic, target_path):
    """
    Saves a dictionary to a pickle file at the specified target path.
    Creates target directory if not existing.
    Parameters
    ----------
    dic : dict
        The dictionary to save to a pickle file.
    target_path : str
        The path of the file where the dictionary shall be stored.
    Returns
    -------
    None
    """
    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(target_path, 'wb') as f:
        pickle.dump(dic, f)


def cmip_plot(ax, df, target, title=None, precip=False, intv_sum='ME', intv_mean='10Y',
              target_label='Target', show_target_label=False, rolling=None):
    """Resamples and plots climate model and target data."""
    if intv_mean == '10Y' or intv_mean == '5Y' or intv_mean == '20Y':
        closure = 'left'
    else:
        closure = 'right'
    if not precip:
        if rolling is not None:
            ax.plot(df.resample(intv_mean, closed=closure, label='left').mean().iloc[:, :].rolling(rolling).mean(), linewidth=1.2)
        else:
            ax.plot(df.resample(intv_mean, closed=closure, label='left').mean().iloc[:, :], linewidth=1.2)
        era_plot, = ax.plot(target['tas'].resample(intv_mean).mean(), linewidth=1.5, c='red', label=target_label,
                            linestyle='dashed')
    else:
        if rolling is not None:
            ax.plot(df.resample(intv_sum, closed=closure, label='left').sum().iloc[:, :].rolling(rolling).mean(), linewidth=1.2)
        else:
            ax.plot(df.resample(intv_sum, closed=closure, label='left').sum().iloc[:, :], linewidth=1.2)
        era_plot, = ax.plot(target['pr'].resample(intv_sum).sum(), linewidth=1.5,
                            c='red', label=target_label, linestyle='dashed')
    if show_target_label:
        ax.legend(handles=[era_plot], loc='upper left')
    ax.set_title(title)
    ax.grid(True)


def cmip_plot_combined(data, target, title=None, precip=False, intv_sum='ME', intv_mean='10Y',
                       target_label='Target', show=False, filename=None, out_dir='./', rolling=None):
    """Combines multiple subplots of climate data in different scenarios before and after bias adjustment.
    Shows target data for comparison"""
    figure, axis = plt.subplots(2, 2, figsize=(12, 12), sharex="col", sharey="all")
    t_kwargs = {'target': target, 'intv_mean': intv_mean, 'target_label': target_label, 'rolling': rolling}
    p_kwargs = {'target': target, 'intv_mean': intv_mean, 'target_label': target_label,
                'intv_sum': intv_sum, 'precip': True, 'rolling': rolling}
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not precip:
        cmip_plot(axis[0, 0], data['SSP2_raw'], show_target_label=True, title='SSP2 raw', **t_kwargs)
        cmip_plot(axis[0, 1], data['SSP2_adjusted'], title='SSP2 adjusted', **t_kwargs)
        cmip_plot(axis[1, 0], data['SSP5_raw'], title='SSP5 raw', **t_kwargs)
        cmip_plot(axis[1, 1], data['SSP5_adjusted'], title='SSP5 adjusted', **t_kwargs)
        figure.legend(data['SSP5_adjusted'].columns, loc='lower right', ncol=6, mode="expand")
        figure.tight_layout()
        figure.subplots_adjust(bottom=0.15, top=0.92)
        figure.suptitle(title, fontweight='bold')
        plt.savefig(out_dir + filename)
        if show:
            plt.show()
    else:
        cmip_plot(axis[0, 0], data['SSP2_raw'], show_target_label=True, title='SSP2 raw', **p_kwargs)
        cmip_plot(axis[0, 1], data['SSP2_adjusted'], title='SSP2 adjusted', **p_kwargs)
        cmip_plot(axis[1, 0], data['SSP5_raw'], title='SSP5 raw', **p_kwargs)
        cmip_plot(axis[1, 1], data['SSP5_adjusted'], title='SSP5 adjusted', **p_kwargs)
        figure.legend(data['SSP5_adjusted'].columns, loc='lower right', ncol=6, mode="expand")
        figure.tight_layout()
        figure.subplots_adjust(bottom=0.15, top=0.92)
        figure.suptitle(title, fontweight='bold')
        plt.savefig(out_dir + filename)
        if show:
            plt.show()


def df2long(df, intv_sum='ME', intv_mean='YE', precip=False):
    """Resamples dataframes and converts them into long format to be passed to seaborn.lineplot()."""

    if precip:
        df = df.resample(intv_sum).sum()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='prec')
    else:
        df = df.resample(intv_mean).mean()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='temp')
    return df


def cmip_plot_ensemble(cmip, target, precip=False, intv_sum='ME', intv_mean='YE', figsize=(10, 6), site_label:str=None,
                       target_label='ERA5L', show=True, out_dir='./', filename='cmip6_ensemble'):
    """
    Plots the multi-model mean of climate scenarios including the 90% confidence interval.
    Parameters
    ----------
    cmip: dict
        A dictionary with keys representing the different CMIP6 models and scenarios as pandas dataframes
        containing data of temperature and/or precipitation.
    target: pandas.DataFrame
        Dataframe containing the historical reanalysis data.
    precip: bool
        If True, plot the mean precipitation. If False, plot the mean temperature. Default is False.
    intv_sum: str
        Interval for precipitation sums. Default is monthly ('ME').
    intv_mean: str
        Interval for the mean of temperature data or precipitation sums. Default is annual ('YE').
    figsize: tuple
        Figure size for the plot. Default is (10,6).
    show: bool
        If True, show the resulting plot. If False, do not show it. Default is True.
    out_dir: str
        Target directory to save figure
    """

    warnings.filterwarnings(action='ignore')
    figure, axis = plt.subplots(figsize=figsize)

    # Define color palette
    colors = ['darkorange', 'orange', 'darkblue', 'dodgerblue']
    # create a new dictionary with the same keys but new values from the list
    col_dict = {key: value for key, value in zip(cmip.keys(), colors)}

    if site_label is None:
        site_label = str()
    else:
        site_label = f'"{site_label}" - '

    if precip:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_sum=intv_sum, intv_mean=intv_mean, precip=True)
            sns.lineplot(data=df, x='TIMESTAMP', y='prec', color=col_dict[i])
        axis.set(xlabel='Year', ylabel='Precipitation [mm]')
        if intv_sum == 'ME':
            figure.suptitle(site_label + 'Ensemble Mean of Monthly Precipitation', fontweight='bold')
        elif intv_sum == 'YE':
            figure.suptitle(site_label + 'Ensemble Mean of Annual Precipitation', fontweight='bold')
        target_plot = axis.plot(target.resample(intv_sum).sum(), linewidth=1.5, c='black',
                                label=target_label, linestyle='dashed')
    else:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_mean=intv_mean)
            sns.lineplot(data=df, x='TIMESTAMP', y='temp', color=col_dict[i])
        axis.set(xlabel='Year', ylabel='Air Temperature [K]')
        if intv_mean == '10Y':
            figure.suptitle(site_label + 'Ensemble Mean of 10y Air Temperature', fontweight='bold')
        elif intv_mean == 'YE':
            figure.suptitle(site_label + 'Ensemble Mean of Annual Air Temperature', fontweight='bold')
        elif intv_mean == 'ME':
            figure.suptitle(site_label + 'Ensemble Mean of Monthly Air Temperature', fontweight='bold')
        target_plot = axis.plot(target.resample(intv_mean).mean(), linewidth=1.5, c='black',
                                label=target_label, linestyle='dashed')
    axis.legend(['SSP2_raw', '_ci1', 'SSP2_adjusted', '_ci2', 'SSP5_raw', '_ci3', 'SSP5_adjusted', '_ci4'],
                loc="upper center", bbox_to_anchor=(0.43, -0.15), ncol=4,
                frameon=False)  # First legend --> Workaround as seaborn lists CIs in legend
    leg = Legend(axis, target_plot, [target_label], loc='upper center', bbox_to_anchor=(0.83, -0.15), ncol=1,
                 frameon=False)  # Second legend (Target)
    axis.add_artist(leg)
    plt.grid()

    figure.tight_layout(rect=[0, 0.02, 1, 1])  # Make some room at the bottom
    if not precip:
        plt.savefig(out_dir + f'{filename}_temperature.png')
    else:
        plt.savefig(out_dir + f'{filename}_precipitation.png')

    if show:
        plt.show()
    warnings.filterwarnings(action='always')


def prob_plot(original, target, corrected, ax, title=None, ylabel="Temperature [K]", **kwargs):
    """
    Combines probability plots of climate model data before and after bias adjustment
    and the target data.

    Parameters
    ----------
    original : pandas.DataFrame
        The original climate model data.
    target : pandas.DataFrame
        The target data.
    corrected : pandas.DataFrame
        The climate model data after bias adjustment.
    ax : matplotlib.axes.Axes
        The axes on which to plot the probability plot.
    title : str, optional
        The title of the plot. Default is None.
    ylabel : str, optional
        The label for the y-axis. Default is "Temperature [K]".
    **kwargs : dict, optional
        Additional keyword arguments passed to the probscale.probplot() function.

    Returns
    -------
    fig : matplotlib Figure
        The generated figure.
    """

    scatter_kws = dict(label="", marker=None, linestyle="-")
    common_opts = dict(plottype="qq", problabel="", datalabel="", **kwargs)

    scatter_kws["label"] = "original"
    fig = probscale.probplot(original, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "target"
    fig = probscale.probplot(target, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "adjusted"
    fig = probscale.probplot(corrected, ax=ax, scatter_kws=scatter_kws, **common_opts)

    ax.set_title(title)

    ax.set_xlabel("Standard Normal Quantiles")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    score = round(target.corr(corrected), 2)
    ax.text(0.05, 0.8, f"R² = {score}", transform=ax.transAxes, fontsize=15)

    return fig


def pp_matrix(original, target, corrected, scenario=None, nrow=7, ncol=5, precip=False,
              starty=1979, endy=2022, show=False, out_dir='./', target_label='ERA5-Land', site:str=None):
    """
    Arranges the prob_plots of all CMIP6 models in a matrix and adds the R² score.
    Parameters
    ----------
    original : pandas.DataFrame
        The original climate model data.
    target : pandas.DataFrame
        The target data.
    corrected : pandas.DataFrame
        The climate model data after bias adjustment.
    scenario : str, optional
        The climate scenario to be added to the plot title.
    nrow : int, optional
        The number of rows in the plot matrix. Default is 7.
    ncol : int, optional
        The number of columns in the plot matrix. Default is 5.
    precip : bool, optional
        Indicates whether the data is precipitation data. Default is False.
    show : bool, optional
        Indicates whether to display the plot. Default is False.
    out_dir: str
        Target directory to save figure
    Returns
    -------
    None
    """

    starty = f'{str(starty)}-01-01'
    endy = f'{str(endy)}-12-31'
    period = slice(starty, endy)

    if site is None:
        site_label = str()
    else:
        site_label = f'"{site}" - '

    if precip:
        var = 'Precipitation'
        var_label = 'Monthly ' + var
        unit = ' [mm]'
        original = original.resample('ME').sum()
        target = target.resample('ME').sum()
        corrected = corrected.resample('ME').sum()
    else:
        var = 'Temperature'
        var_label = 'Daily Mean ' + var
        unit = ' [K]'

    fig = plt.figure(figsize=(16, 16))

    for i, col in enumerate(original.columns):
        ax = plt.subplot(nrow, ncol, i + 1)
        prob_plot(original[col][period], target[period],
                  corrected[col][period], ax=ax, ylabel=var + unit)
        ax.set_title(col, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ['original (CMIP6 raw)', f'target ({target_label})', 'adjusted (CMIP6 after SDM)'], loc='lower right',
               bbox_to_anchor=(0.96, 0.024), fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.7, wspace=0.4)
    starty = period.start.split('-')[0]
    endy = period.stop.split('-')[0]
    if scenario is None:
        fig.suptitle(site_label + f'Probability Plots of CMIP6 and {target_label} ' + var_label + ' (' + starty + '-' + endy + ')',
                     fontweight='bold', fontsize=20)
    else:
        fig.suptitle(site_label + 'Probability Plots of CMIP6 (' + scenario + f') and {target_label} ' + var_label, fontweight='bold', fontsize=20)
    plt.subplots_adjust(top=0.93)
    if precip:
        plt.savefig(out_dir + f'cmip6_ensemble_probplots_{site}_precipitation_{scenario}.png')
    else:
        plt.savefig(out_dir + f'cmip6_ensemble_probplots_{site}_temperature_{scenario}.png')

    if show:
        plt.show()


def write_output(adj_dict: dict, output: str, station: str, starty: str, endy: str, type: str, ndigits: int=3):
    """
    Writes the full output of the adjusted dictionary to CSV files. Variable name is determined based on the mean value
    of the first DataFrame.
    Parameters
    ----------
    adj_dict : dict
        A dictionary containing DataFrames to be processed.
    output : str
        Output directory path.
    station : str
        Station name for naming convention.
    starty : str
        Start year for naming convention.
    endy : str
        End year for naming convention.
    type : str
        Type of the file written. Usually 'full' or 'summary'.
    ndigits : int
        Number of digits the dataframes will be rounded to before writing them to file.
    Returns
    -------
    None
    """
    if adj_dict['SSP2_raw'].mean().mean() > 100:
        var = 'temp'
    else:
        var = 'prec'
    final_output = f'{output}/posterior/{type}/'
    if not os.path.exists(final_output):
        os.makedirs(final_output)
    for name, data in adj_dict.items():
        round(data, ndigits).to_csv(f'{final_output}{name}_{station}_{type}_{var}_{starty}-{endy}.csv')


def ensemble_summary(df):
    """
    Calculate ensemble summary statistics for the input DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with ensemble member data.
    Returns
    -------
    pandas.DataFrame
        DataFrame with ensemble summary statistics including mean, median, min, max, standard deviation,
        and 90% confidence interval bounds.
    """
    summary_df = pd.DataFrame(index=df.index)
    # Calculate ensemble mean
    summary_df['ensemble_mean'] = df.mean(axis=1)
    # Calculate ensemble median
    summary_df['ensemble_median'] = df.median(axis=1)
    # Calculate ensemble min
    summary_df['ensemble_min'] = df.min(axis=1)
    # Calculate ensemble max
    summary_df['ensemble_max'] = df.max(axis=1)
    # Calculate ensemble standard deviation
    summary_df['ensemble_sd'] = df.std(axis=1)
    # Calculate ensemble 90% confidence interval lower bound
    summary_df['ensemble_90perc_CI_lower'] = df.quantile(0.05, axis=1)
    # Calculate ensemble 90% confidence interval upper bound
    summary_df['ensemble_90perc_CI_upper'] = df.quantile(0.95, axis=1)

    return summary_df


def summary_dict(results_dict: dict):
    """
    Generate a dictionary of ensemble summary DataFrames for each value in the input dictionary.
    Parameters
    ----------
    results_dict : dict
        Input dictionary containing ensemble results.
    Returns
    -------
    dict
        A summary dictionary with keys as original keys and values as ensemble summaries.
    """
    return {key: ensemble_summary(value) for key, value in results_dict.items()}


class ClimateScenarios:
    def __init__(self, output, reanalysis_dir, polygon_path, gee_project='matilda-edu', download=False, load_backup=False, show=True,
                 starty=1979, endy=2100, processes=5, plots=True):
        self.polygon_path = polygon_path
        self.poly_name = os.path.splitext(os.path.basename(polygon_path))[0]
        self.output = os.path.join(output, self.poly_name)
        self.gee_project = gee_project
        self.download = download
        self.load_backup = load_backup
        self.show = show
        self.starty = starty
        self.endy = endy
        self.processes = processes
        self.reanalysis_dir = reanalysis_dir
        self.plots = plots

    def cmip6_data_processing(self):
        cmip_dir = os.path.join(self.output, 'raw/')
        self.cmip6_polygon = CMIP6PolygonProcessor(self.polygon_path, self.gee_project, self.starty, self.endy,
                                                   cmip_dir, self.reanalysis_dir, self.processes)
        print(f'CMIP6PolygonProcessor instance for polygon "{self.poly_name}" configured.')

        if self.download:
            self.cmip6_polygon.download_cmip6_data()

        self.cmip6_polygon.bias_adjustment()
        self.temp_cmip = self.cmip6_polygon.ssp_tas_dict
        self.prec_cmip = self.cmip6_polygon.ssp_pr_dict
        self.training_dict = self.cmip6_polygon.training_dict

    def data_checks(self):
        self.temp_cmip, self.prec_cmip = apply_filters(self.temp_cmip, self.prec_cmip, zscore_threshold=3,
                                                       jump_threshold=5, resampling_rate='YE')
        print(f'Consistency-checks applied to adjusted data for "{self.poly_name}".')

        process_nested_dict(self.temp_cmip, round, ndigits=3)
        process_nested_dict(self.prec_cmip, round, ndigits=3)
        print(f'Data for "{self.poly_name}" rounded to save storage space.')

    def create_plots(self):
        # Load CHELSA data as "target" for comparison
        tas_chelsa = pd.read_csv(os.path.join(self.reanalysis_dir, f'{self.poly_name}_tas.csv'), index_col='time',
                                 parse_dates=True)
        pr_chelsa = pd.read_csv(os.path.join(self.reanalysis_dir, f'{self.poly_name}_pr.csv'), index_col='time',
                                parse_dates=True)

        plot_dir = os.path.join(self.output, 'Plots/')

        cmip_plot_combined(data=self.temp_cmip, target=tas_chelsa,
                           title=f'"{self.poly_name}" - 5y Rolling Mean of Annual Air Temperature',
                           target_label='CHELSA-W5E5',
                           filename=f'cmip6_bias_adjustment_{self.poly_name}_temperature.png', show=self.show,
                           intv_mean='YE', rolling=5, out_dir=plot_dir)
        cmip_plot_combined(data=self.prec_cmip, target=pr_chelsa.dropna(),
                           title=f'"{self.poly_name}" - 5y Rolling Mean of Annual Precipitation',
                           precip=True,
                           target_label='CHELSA-W5E5',
                           filename=f'cmip6_bias_adjustment_{self.poly_name}_precipitation.png', show=self.show,
                           intv_sum='YE', rolling=5, out_dir=plot_dir)
        print(f'Figures for CMIP6 bias adjustment for "{self.poly_name}" created.')

        cmip_plot_ensemble(self.temp_cmip, tas_chelsa['tas'], intv_mean='YE', show=self.show,
                           out_dir=plot_dir, target_label="CHELSA-W5E5",
                           filename=f'cmip6_ensemble_{self.poly_name}', site_label=self.poly_name)
        cmip_plot_ensemble(self.prec_cmip, pr_chelsa['pr'].dropna(), precip=True, intv_sum='YE', show=self.show,
                           out_dir=plot_dir, target_label="CHELSA-W5E5", site_label=self.poly_name,
                           filename=f'cmip6_ensemble_{self.poly_name}')
        print(f'Figures for CMIP6 ensembles for "{self.poly_name}" created.')

        start_temp = tas_chelsa.first_valid_index().year
        end_temp = tas_chelsa.last_valid_index().year
        start_prec = pr_chelsa.dropna().first_valid_index().year
        end_prec = pr_chelsa.dropna().last_valid_index().year

        pp_matrix(self.training_dict['tas']['SSP2_raw_train'], tas_chelsa['tas'], self.training_dict['tas']['SSP2_adjusted_train'], scenario='SSP2',
                  starty=start_temp, endy=end_temp, target_label='CHELSA-W5E5', out_dir=plot_dir,
                  show=self.show, site=self.poly_name)
        pp_matrix(self.training_dict['tas']['SSP5_raw_train'], tas_chelsa['tas'], self.training_dict['tas']['SSP5_adjusted_train'], scenario='SSP5',
                  starty=start_temp, endy=end_temp, target_label='CHELSA-W5E5', out_dir=plot_dir,
                  show=self.show, site=self.poly_name)

        pp_matrix(self.training_dict['pr']['SSP2_raw_train'], pr_chelsa['pr'].dropna().astype(float),
                  self.training_dict['pr']['SSP2_adjusted_train'], precip=True, starty=start_prec, endy=end_prec,
                  target_label='CHELSA-W5E5', scenario='SSP2', out_dir=plot_dir,
                  show=self.show, site=self.poly_name)
        pp_matrix(self.training_dict['pr']['SSP5_raw_train'], pr_chelsa['pr'].dropna().astype(float),
                  self.training_dict['pr']['SSP5_adjusted_train'], precip=True, starty=start_prec, endy=end_prec,
                  target_label='CHELSA-W5E5', scenario='SSP5', out_dir=plot_dir,
                  show=self.show, site=self.poly_name)
        print(f'Figures for CMIP6 bias adjustment performance for "{self.poly_name}" created.')

        plt.close('all')

    def write_output_files(self):
        write_output(self.prec_cmip, self.output, self.poly_name, self.starty, self.endy, type='full')
        write_output(self.temp_cmip, self.output, self.poly_name, self.starty, self.endy, type='full')

        self.temp_summary = summary_dict(self.temp_cmip)
        self.prec_summary = summary_dict(self.prec_cmip)
        write_output(self.temp_summary, self.output, self.poly_name, self.starty, self.endy, type='summary')
        write_output(self.prec_summary, self.output, self.poly_name, self.starty, self.endy, type='summary')

        self.ensemble_mean_temp = pd.concat(
            [df['ensemble_mean'] for key, df in self.temp_summary.items()
             if 'adjusted' in key and 'ensemble_mean' in df.columns],
            axis=1
        ).mean(axis=1).to_frame(name='ensemble_mean')

        self.ensemble_mean_prec = pd.concat(
            [df['ensemble_mean'] for key, df in self.prec_summary.items()
             if 'adjusted' in key and 'ensemble_mean' in df.columns],
            axis=1
        ).mean(axis=1).to_frame(name='ensemble_mean')

        print(f'Output files for "{self.poly_name}" written.')

    def write_dat_files(self):

        def format_decimal_time(date):
            year = date.year
            start_of_year = pd.Timestamp(f'{year}-01-01')
            decimal_time = year + (date - start_of_year).days / 365.25
            return decimal_time

        def save_to_dat(temp_df, prec_df, scenario, ensemble_member):
            # Ensure the index is a DatetimeIndex
            if not isinstance(temp_df.index, pd.DatetimeIndex):
                temp_df.index = pd.to_datetime(temp_df.index)
            if not isinstance(prec_df.index, pd.DatetimeIndex):
                prec_df.index = pd.to_datetime(prec_df.index)

            # Convert Series to DataFrame
            if isinstance(temp_df, pd.Series):
                temp_df = temp_df.to_frame(name=ensemble_member)
            if isinstance(prec_df, pd.Series):
                prec_df = prec_df.to_frame(name=ensemble_member)

            # Separate the pre-2021 data (CHELSA period)
            pre_2021 = temp_df.index < pd.Timestamp('2021-01-01')

            # --- HOTFIX: CHELSA only has data until 2020 ---

            # Extract the ensemble mean data
            temp_series = self.ensemble_mean_temp.iloc[:, 0]
            prec_series = self.ensemble_mean_prec.iloc[:, 0]

            # Replace values before 2021 with ensemble mean
            temp_df.loc[pre_2021, ensemble_member] = temp_series.loc[pre_2021]
            prec_df.loc[pre_2021, ensemble_member] = prec_series.loc[pre_2021]

            # Convert temperature from Kelvin to Celsius
            temp_df[ensemble_member] = temp_df[ensemble_member] - 273.15

            # Combine temperature and precipitation data
            combined_df = temp_df.join(prec_df, lsuffix='_temp', rsuffix='_prec')

            combined_df['Year'] = combined_df.index.year
            combined_df['Month'] = combined_df.index.month
            combined_df['DOY'] = combined_df.index.dayofyear
            combined_df['decimal.time'] = combined_df.index.map(format_decimal_time)
            combined_df['temp(degC)'] = combined_df[f'{ensemble_member}_temp']
            combined_df['prec(mm/day)'] = combined_df[f'{ensemble_member}_prec']

            columns = ['Year', 'Month', 'DOY', 'decimal.time', 'temp(degC)', 'prec(mm/day)']
            combined_df = combined_df[columns]

            output_dir = os.path.join(self.output, 'dat_files')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{ensemble_member}_{scenario}.dat')

            with open(output_file, 'w') as f:
                f.write(f'Future meteorological forcing for glacier {self.poly_name}\n')
                f.write(f'{ensemble_member}/{scenario}\n')
                f.write('Year  Month  DOY  decimal.time  temp(degC)  prec(mm/day)\n')
                combined_df.to_csv(f, sep=' ', index=False, header=False, float_format='%.4f')

        for scenario, temp_data_dict in {'SSP2': self.temp_cmip['SSP2_adjusted'],
                                         'SSP5': self.temp_cmip['SSP5_adjusted']}.items():
            for ensemble_member, temp_df in tqdm(temp_data_dict.items(), desc=f'{scenario} GCMs', unit=' GCMs'):
                prec_df = self.prec_cmip[scenario + '_adjusted'][ensemble_member]
                save_to_dat(temp_df, prec_df, scenario, ensemble_member)

        print(f'Output .dat files for "{self.poly_name}" written.')

    def complete_workflow(self):
        if self.load_backup:
            self.temp_cmip = pickle_to_dict(
                os.path.join(self.output, f'back_ups/temp_{self.poly_name}_adjusted.pickle'))
            self.prec_cmip = pickle_to_dict(
                os.path.join(self.output, f'back_ups/prec_{self.poly_name}_adjusted.pickle'))
            self.training_dict = pickle_to_dict(os.path.join(self.output, f'back_ups/training_{self.poly_name}.pickle'))
            print(f'Back-up of adjusted data for "{self.poly_name}" found and loaded.')

        else:
            self.cmip6_data_processing()
            self.data_checks()
            dict_to_pickle(self.temp_cmip,
                           os.path.join(self.output, f'back_ups/temp_{self.poly_name}_adjusted.pickle'))
            dict_to_pickle(self.prec_cmip,
                           os.path.join(self.output, f'back_ups/prec_{self.poly_name}_adjusted.pickle'))
            dict_to_pickle(self.training_dict,
                           os.path.join(self.output, f'back_ups/training_{self.poly_name}.pickle'))
            print(f'Back-up of adjusted data for "{self.poly_name}" written.')

        if self.plots:
            self.create_plots()

        self.write_output_files()
        self.write_dat_files()

        print(f'Finished workflow for polygon "{self.poly_name}".\n--------------------------------------')


## Manual run for debugging

import configparser

config = configparser.ConfigParser()
config.read('/home/phillip/Seafile/EBA-CA/Repositories/chelsa-nex/config.ini')
settings = config['settings']

# Extract settings from the configuration file
chelsa_dir = settings.get('chelsa_dir')
cmip_dir = settings.get('output_dir')
polygon_path = settings.get('polygon')
gee_project = settings.get('gee_project')
download = settings.getboolean('download')
show = settings.getboolean('show')
load_backup = settings.getboolean('load_backup')
processes = settings.getint('processes')
starty = settings.getint('start_year')
endy = settings.getint('end_year')


# instance = ClimateScenarios(output=cmip_dir,
#                             reanalysis_dir=chelsa_dir,
#                             polygon_path=polygon_path,
#                             gee_project=gee_project,
#                             download=True,
#                             load_backup=False,
#                             show=show,
#                             starty=starty,
#                             endy=endy,
#                             processes=processes,
#                             plots=True)
#
# instance.complete_workflow()

instance = ClimateScenarios(output='/home/phillip/Seafile/EBA-CA/Repositories/chelsa-nex/debugDir',
                            reanalysis_dir=chelsa_dir,
                            polygon_path=polygon_path,
                            gee_project=gee_project,
                            download=True,
                            load_backup=False,
                            show=show,
                            starty=starty,
                            endy=endy,
                            processes=processes,
                            plots=False)

instance.complete_workflow()

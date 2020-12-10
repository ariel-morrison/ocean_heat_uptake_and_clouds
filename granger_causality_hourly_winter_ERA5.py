import sys
import os
import numpy as np
import numpy.ma as ma
import xarray as xr
import pandas as pd
import warnings
from global_land_mask import globe
# ----------------------------------------

working_dir = '/global/cscratch1/sd/armorris/'
os.chdir(working_dir)

latitude = np.linspace(-90.0, -30.0, 241)
longitude = np.linspace(-180.0, 179.75, 1440)


def landMask(lat, lon):
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    is_on_land = globe.is_land(lat_grid, lon_grid)
    return is_on_land


def lag_linregress_3D_range(x, y, lagx=0, lagy=0):
    x, y = xr.align(x, y)

    if lagx != 0:
        # If x lags y by 1, x must be shifted 1 step backwards.
        # Works with negative and positive lags
        # E.g., if lag = -1, x is shifted 1 step backwards
        # If lag = 2, x is shifted 2 steps forwards
        x    = x.shift(time=lagx).dropna(dim='time')
        x, y = xr.align(x, y)

    if lagy != 0:
        y   = y.shift(time = lagy).dropna(dim='time')
        x, y = xr.align(x,y)

    # 3. Compute data length, mean and standard deviation along time axis for further use:
    n     = x.shape[0]
    xmean = x.mean(axis=0, skipna=True)
    ymean = y.mean(axis=0, skipna=True)
    xstd  = x.std(axis=0, skipna=True)
    ystd  = y.std(axis=0, skipna=True)

    # 4. Compute covariance along time axis
    cov   =  np.nansum((x - xmean)*(y - ymean), axis=0)/(n-1)

    # 5. Compute correlation along time axis
    cor   = cov/(xstd*ystd)

    return cor


def open_reanalysis_data(working_dir, year):
    os.chdir(working_dir)

    latitude = np.linspace(-90.0, -30.0, 241)
    longitude = np.linspace(-180.0, 179.75, 1440)

    # SEA ICE
    ds = xr.open_dataset(working_dir + '/ERA-5/data/hourly_data_correlations/siconc_hourly_winter_' + str(year) + '.nc')
    seaIce = ds.siconc
    ds.close()

    # CLOUDS
    # Total clouds = 'tcc'
    # Low clouds = 'lcc'
    ds = xr.open_dataset(working_dir + '/ERA-5/data/hourly_data_correlations/clouds_hourly_winter_' + str(year) + '.nc')
    tcc = ds.tcc
    lcc = ds.lcc

    # OCEAN HEAT UPTAKE
    ds = xr.open_dataset(working_dir + '/ERA-5/data/hourly_data_correlations/southOceanOHU_hourly_winter_' + str(year) + '.nc')
    sohu = ds.sohu
    ds.close()

    # STABILITY
    ds = xr.open_dataset(working_dir + '/ERA-5/data/hourly_data_correlations/stability_800hPa_hourly_winter_' + str(year) + '.nc')
    stability = ds.stability800
    ds.close()

    # TURBULENT HEAT FLUXES
    ds = xr.open_dataset(working_dir + '/ERA-5/data/hourly_data_correlations/thflx_hourly_winter_' + str(year) + '.nc')
    thflx = ds.thflx
    ds.close()
    ds = xr.open_dataset(working_dir +
                         '/ERA-5/data/hourly_data_correlations/turbulent_fluxes_hourly_winter_' + str(year) + '.nc')
    shflx = ds.shflx
    lhflx = ds.lhflx
    ds.close()

    # TROPOSPHERIC TEMPERATURE AT 800mb
    ds = xr.open_dataset(working_dir + '/ERA-5/data/hourly_data_correlations/temp_profile_800hPa_hourly_winter_' + str(year) + '.nc')
    temp800 = ds.t
    ds.close()

    return latitude, longitude, seaIce, tcc, lcc, sohu, stability, thflx, shflx, lhflx, temp800


def make_zonal_means(working_dir, year, latitude, longitude, seaIce, seaIce_cutoff, tcc, lcc, sohu, stability, thflx, shflx, lhflx, temp800):
    latitude, longitude, seaIce, tcc, lcc, sohu, stability, thflx, shflx, lhflx, temp800 = open_reanalysis_data(working_dir, year)

    latmin = -65
    latmax = -44.75

    # Constrain zonal means to 45-65S:
    latmin_ind = np.abs(latmin-latitude).argmin()
    latmax_ind = np.abs(latmax-latitude).argmin()

    is_on_land = landMask(latitude, longitude)

    # Cosine latitude weights for zonal averaging:
    weights = np.cos(np.deg2rad(latitude[latmin_ind:latmax_ind]))

    # Mask data by land and sea ice
    tcc_for_hov = tcc.where(is_on_land == False).where(seaIce < seaIce_cutoff)
    lcc_for_hov = lcc.where(is_on_land == False).where(seaIce < seaIce_cutoff)
    sohu_for_hov = sohu.where(is_on_land == False).where(seaIce < seaIce_cutoff)
    stability_for_hov = stability.where(is_on_land == False).where(seaIce < seaIce_cutoff)
    thflx_for_hov = thflx.where(is_on_land == False).where(seaIce < seaIce_cutoff)
    lhflx_for_hov = lhflx.where(is_on_land == False).where(seaIce < seaIce_cutoff)
    temp800_for_hov = temp800.where(is_on_land == False).where(seaIce < seaIce_cutoff)

    # Calculate zonal means to get 2D array of (time x longitude)
    tcc_lon = (tcc_for_hov[:, latmin_ind:latmax_ind, :] * weights[None, :, None]).sum(axis=1, skipna=True) / np.nansum(weights)
    lcc_lon = (lcc_for_hov[:, latmin_ind:latmax_ind, :] * weights[None, :, None]).sum(axis=1, skipna=True) / np.nansum(weights)
    sohu_lon = (sohu_for_hov[:, latmin_ind:latmax_ind, :] * weights[None, :, None]).sum(axis=1, skipna=True) / np.nansum(weights)
    stability_lon = (stability_for_hov[:, latmin_ind:latmax_ind, :] * weights[None, :, None]).sum(axis=1, skipna=True) / np.nansum(weights)
    thflx_lon = (thflx_for_hov[:, latmin_ind:latmax_ind, :] * weights[None, :, None]).sum(axis=1, skipna=True) / np.nansum(weights)
    lhflx_lon = (lhflx_for_hov[:, latmin_ind:latmax_ind, :] * weights[None, :, None]).sum(axis=1, skipna=True) / np.nansum(weights)
    temp800_lon = (temp800_for_hov[:, latmin_ind:latmax_ind, :] * weights[None, :, None]).sum(axis=1, skipna=True) / np.nansum(weights)

    # Test stationarity of timeseries
    # # Make time series stationary by differencing n+1 from n
    # lcc_stat = lcc_lon.diff('time', n=1)
    # sohu_stat = sohu_lon.diff('time', n=1)
    # thflx_stat = thflx_lon.diff('time', n=1)
    # stability_stat = stability_lon.diff('time', n=1)
    # tcc_stat = tcc_lon.diff('time', n=1)
    # temp800_stat = temp800_lon.diff('time', n=1)

    return lcc_lon, tcc_lon, stability_lon, sohu_lon, thflx_lon, lhflx_lon, temp800_lon


def granger_timeseries(working_dir, year, latitude, longitude, seaIce_cutoff, test_stationarity):
    latitude, longitude, seaIce, tcc, lcc, sohu, stability, thflx, shflx, lhflx, temp800 = open_reanalysis_data(working_dir, year)

    is_on_land = landMask(latitude, longitude)
    seaIceMask = seaIce.groupby('time.month').mean(dim='time').mean(dim='month').fillna(1)
    seaIceMask = seaIceMask > seaIce_cutoff

    deg2rad = np.pi / 180.
    coslat = np.cos(deg2rad * latitude)

    lon2, lat2 = np.meshgrid(longitude, latitude)
    lat_weights = ma.array(np.cos(np.deg2rad(lat2)), mask = seaIceMask)

    latmin = -65
    latmax = -44.75
    latmin_ind = int(np.abs(latmin-latitude).argmin())
    latmax_ind = int(np.abs(latmax-latitude).argmin())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Mask out sea ice pack from each parameter
        # Only interested in heat fluxes/clouds over the ocean
        seaIce_cutoff = np.float(seaIce_cutoff)
        tcc = tcc.where(is_on_land == False).where(seaIce < seaIce_cutoff)
        lcc = lcc.where(is_on_land == False).where(seaIce < seaIce_cutoff)
        sohu = sohu.where(is_on_land == False).where(seaIce < seaIce_cutoff)
        stability = stability.where(is_on_land == False).where(seaIce < seaIce_cutoff)
        thflx = thflx.where(is_on_land == False).where(seaIce < seaIce_cutoff)
        shflx = shflx.where(is_on_land == False).where(seaIce < seaIce_cutoff)
        lhflx = lhflx.where(is_on_land == False).where(seaIce < seaIce_cutoff)
        temp800 = temp800.where(is_on_land == False).where(seaIce < seaIce_cutoff)

        # time series averaged across entire donut
        tcc_SO_timeseries = np.nansum((np.nanmean(tcc[:, latmin_ind:latmax_ind, :], axis=2) * coslat[latmin_ind:latmax_ind]), axis=1)/np.nansum(np.nanmean(lat_weights[latmin_ind:latmax_ind, :], axis=1), axis=0)
        lcc_SO_timeseries = np.nansum((np.nanmean(lcc[:, latmin_ind:latmax_ind, :], axis=2) * coslat[latmin_ind:latmax_ind]), axis=1)/np.nansum(np.nanmean(lat_weights[latmin_ind:latmax_ind, :], axis=1), axis=0)
        sohu_SO_timeseries = np.nansum((np.nanmean(sohu[:, latmin_ind:latmax_ind, :], axis=2) * coslat[latmin_ind:latmax_ind]), axis=1)/np.nansum(np.nanmean(lat_weights[latmin_ind:latmax_ind, :], axis=1), axis=0)
        stability_SO_timeseries = np.nansum((np.nanmean(stability[:, latmin_ind:latmax_ind, :], axis=2)*coslat[latmin_ind:latmax_ind]), axis=1)/np.nansum(np.nanmean(lat_weights[latmin_ind:latmax_ind, :], axis=1), axis=0)
        thflx_SO_timeseries = np.nansum((np.nanmean(thflx[:, latmin_ind:latmax_ind, :], axis=2) * coslat[latmin_ind:latmax_ind]), axis=1)/np.nansum(np.nanmean(lat_weights[latmin_ind:latmax_ind, :], axis=1), axis=0)
        shflx_SO_timeseries = np.nansum((np.nanmean(shflx[:, latmin_ind:latmax_ind, :], axis=2) * coslat[latmin_ind:latmax_ind]), axis=1)/np.nansum(np.nanmean(lat_weights[latmin_ind:latmax_ind, :], axis=1), axis=0)
        lhflx_SO_timeseries = np.nansum((np.nanmean(lhflx[:, latmin_ind:latmax_ind, :], axis=2) * coslat[latmin_ind:latmax_ind]), axis=1)/np.nansum(np.nanmean(lat_weights[latmin_ind:latmax_ind, :], axis=1), axis=0)


        ### ------- Test for stationarity ---------
        # Are the time series values independent of time, or do they show a trend/seasonality?
        # Use an Augmented Dickey-Fuller unit root test:
        # A unit root test determines how strongly a time series is defined by a trend.
        #
        # Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root,
        #     meaning it is non-stationary. It has some time dependent structure.
        # Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does
        #     not have a unit root, meaning it is stationary. It does not have time-dependent structure.
        ### -----------------------------------------
        from statsmodels.tsa.stattools import adfuller

        # longitude bands
        lcc_lon, tcc_lon, stability_lon, sohu_lon, thflx_lon, lhflx_lon, temp800_lon = make_zonal_means(working_dir, year, latitude, longitude, seaIce, seaIce_cutoff, tcc, lcc, sohu, stability, thflx, shflx, lhflx, temp800)

        if test_stationarity == True:
            # longitude bands
            timeseries_lon = [lcc_lon, tcc_lon, stability_lon, sohu_lon, thflx_lon]
            timeseries_lon_name = ['Low clouds', 'Total clouds', 'Stability', 'SOHU', 'Total heat flux']

            # entire donut
            timeseries = [tcc_SO_timeseries, lcc_SO_timeseries, sohu_SO_timeseries, stability_SO_timeseries,
                            thflx_SO_timeseries, shflx_SO_timeseries, lhflx_SO_timeseries]
            timeseriesName = ['Total clouds', 'Low clouds', 'SOHU', 'Stability', 'Total heat flux','Sensible heat flux', 'Latent heat flux']

            print("Testing stationarity of time series. Null hypothesis is non-stationarity: time series has time-dependent structure.")
            for ts in range(0, len(timeseries_lon)):
                for ilon in range(0, len(longitude)):
                    result = adfuller(timeseries_lon[ts][:, ilon])
                    if result[1] > 0.05 and result[0] < result[4]['5%']:
                        print("Null hypothesis NOT rejected. " + str(timeseries_lon_name[ts]) + " at " + str(longitude[ilon]) + " (" + str(year) + ") " + " has time-dependent structure. p-value: %f" % result[1])

            for ts in range(0, len(timeseries)):
                result = adfuller(timeseries[ts])
                if result[1] <= 0.05:
                    print("Null hypothesis rejected. " + str(timeseriesName[ts]) + " (" + str(year) + ") " + " time series is stationary. p-value: %f" % result[1])
                elif result[1] > 0.05 and result[1] < 0.1 and result[0] < result[4]['5%']:
                    print("Null hypothesis rejected. " + str(timeseriesName[ts]) + " (" + str(year) + ") " + " time series is stationary, but " + "p-value: %f" % result[1])
                else:
                    print("Null hypothesis NOT rejected. " + str(timeseriesName[ts]) + " (" + str(year) + ") " + " time series has time-dependent structure. p-value: %f" % result[1])

            del timeseries_lon, timeseries_lon_name, timeseries, timeseriesName, tcc, lcc, sohu, stability, thflx, shflx, lhflx, temp800

        # entire donut
        sohu_gc = pd.DataFrame(data=(sohu_SO_timeseries), columns=['SOHU'])
        tcc_gc = pd.DataFrame(data=(tcc_SO_timeseries), columns=['Clouds'])
        lcc_gc = pd.DataFrame(data=(lcc_SO_timeseries), columns=['Low_clouds'])
        stability_gc = pd.DataFrame(data=(stability_SO_timeseries), columns=['Stability'])
        thflx_gc = pd.DataFrame(data=(thflx_SO_timeseries), columns=['Heat_flux'])
        shflx_gc = pd.DataFrame(data=(shflx_SO_timeseries), columns=['Sensible_heat'])
        lhflx_gc = pd.DataFrame(data=(lhflx_SO_timeseries), columns=['Latent_heat'])

    return sohu_lon, tcc_lon, lcc_lon, stability_lon, thflx_lon, sohu_gc, tcc_gc, lcc_gc, stability_gc, thflx_gc, shflx_gc, lhflx_gc


def granger_causality(working_dir, year, latitude, longitude, seaIce_cutoff, test_stationarity):
    seaIce_cutoff = np.float(seaIce_cutoff)

    sohu_lon, tcc_lon, lcc_lon, stability_lon, thflx_lon, sohu_gc, tcc_gc, lcc_gc, stability_gc, thflx_gc, shflx_gc, lhflx_gc = granger_timeseries(working_dir, year, latitude, longitude, seaIce_cutoff, test_stationarity)


    ### ---------------- Granger causality: ---------------
    # It accepts a 2D array with 2 columns as the main argument.
    # The values are in the first column and the predictor (X) is in the second column.
    # The Null hypothesis is: the series in the second column does not Granger cause the series in the first.
    # If the P-Values are less than a significance level (0.05) then you reject the null hypothesis and conclude
    #       that the said lag of X is indeed useful.
    # The second argument maxlag says till how many lags of Y should be included in the test.
    ### -----------------------------------------------------

    from statsmodels.tsa.stattools import grangercausalitytests

    # longitude bands (averaged over latitude)
    def granger_function(df_lon):
        res = grangercausalitytests(df_lon[['SOHU','Clouds']], maxlag=12, verbose=False)
        out = [np.round(res[key][0]['ssr_ftest'][1], decimals=3) for key in res]
        return pd.Series(out)

    pval_df = pd.DataFrame()

    tcc_lon_df = tcc_lon.to_dataframe()
    tcc_lon_df.columns = ['Clouds']
    sohu_lon_df = sohu_lon.to_dataframe()
    sohu_lon_df.columns = ['SOHU']
    thflx_lon_df= thflx_lon.to_dataframe()
    thflx_lon_df.columns = ['Heat_Flux']
    stability_lon_df = stability_lon.to_dataframe()
    stability_lon_df.columns = ['Stability']

    df_lon = pd.concat([sohu_lon_df, tcc_lon_df], axis=1)
    df_lon = df_lon.reset_index()
    pval_df = df_lon.groupby('longitude').apply(granger_function)
    pval_df.to_csv('lon_bands_clouds_predict_sohu_granger_causality_ERA5_winter_' + str(year) + '.csv')
    np.save('lon_bands_clouds_predict_sohu_granger_causality_ERA5_winter_' + str(year) + '.npy', pval_df)


    # entire donut or longitude sections
    df = pd.concat([sohu_gc, tcc_gc, lcc_gc, stability_gc, thflx_gc, shflx_gc, lhflx_gc], axis=1)
    colName = list(df.columns)
    pval_df = pd.DataFrame()

    # maxlag = how many time steps to shift X back
    # find the p-value at each lag for how well X predicts Y
    # values (Y)    = first column
    # predictor (X) = second column
    for predictand in range(0, len(list(df.columns))):
        for predictor in range(0, len(list(df.columns))):
            res = grangercausalitytests(df[[str(colName[predictand]), str(colName[predictor])]], maxlag=12, verbose=False)
            out = [res[key][0]['ssr_ftest'] for key in res]
            pval = np.zeros((12))
            for lag in range(0, 12): pval[lag] = np.round(out[lag][1], decimals=3)
            pval = pd.DataFrame(pval)
            pval.columns = [str(colName[predictor]) + '_predicts_' + str(colName[predictand])]
            pval_df = pd.concat([pval_df, pval], axis=1)

    pval_df.to_csv('granger_causality_ERA5_winter_' + str(year) + '.csv')

    return


if __name__=='__main__':
    working_dir, year, latitude, longitude, seaIce_cutoff, test_stationarity = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
    granger_causality(working_dir, year, latitude, longitude, seaIce_cutoff, test_stationarity)

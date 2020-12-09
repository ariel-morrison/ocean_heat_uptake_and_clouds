import sys
import os
import numpy as np
import numpy.ma as ma
import xarray
import xarray as xr
from scipy import stats
from scipy import signal
import warnings
from global_land_mask import globe

working_dir = '/global/cscratch1/sd/armorris/'
os.chdir(working_dir)

def landMask(lat,lon):
    lon_grid, lat_grid = np.meshgrid(lon,lat)
    is_on_land = globe.is_land(lat_grid,lon_grid)
    return is_on_land

def read_reanalysis_data(working_dir):
    # Reanalysis data (heat fluxes, pressure, SST, winds)
    dsERA = xr.open_dataset(working_dir + '/varForOHU_ERA5.nc')
    latitude = dsERA.latitude
    longitude = dsERA.longitude
    shflx = dsERA.msshf
    lhflx = dsERA.mslhf
    sfc_lw_net = dsERA.msnlwrf
    sfc_sw_net = dsERA.msnswrf
    sfc_sw_net_clr = dsERA.msnswrfcs
    sfc_lw_net_clr = dsERA.msnlwrfcs
    dsERA.close()

    ds = xr.open_dataset(working_dir + '/varSurface_ERA5.nc')
    windSpeed = ds.si10
    seaIce = ds.siconc
    mslp = ds.msl
    msst = ds.sst
    ds.close()

    ds = xr.open_dataset(working_dir + '/varClouds_ERA5.nc')
    tcc = ds.tcc
    lcc = ds.lcc
    mcc = ds.mcc
    hcc = ds.hcc
    tclwp = ds.tclw
    ds.close()

    ## tclwp per cloud fraction
    # get to cloud % instead of fraction
    tcc_for_tclwp = totalCloud * 100.
    # get to g/m3
    tclwp_for_tclwp_per_tcc = tclwp * 1000.
    tclwp_per_tcc = tclwp_for_tclwp_per_tcc / tcc_for_tclwp
    tclwp_per_tcc = tclwp_per_tcc.where(tcc_for_tclwp > 1)

    # stability
    ds = xr.open_dataset(working_dir + '/ERA-5/data/stability_800hPa_monthly_ERA5_1979_2019.nc')
    stability = ds.__xarray_dataarray_variable__
    ds.close()

    # temperature profile
    ds = xr.open_dataset(os.path.join(working_dir + 'ERA-5/data/temp_profile_650_to_1000hPa_1979_2019.nc'))
    ds = ds.sel(latitude=slice(None, None, -1))
    temp = ds.t
    temp_zonal_mean = temp.mean(axis=3).rename('temp_zonal_mean')
    ds.close()

    dsERA = xr.open_dataset(working_dir + '/varProfiles_ERA5.nc')
    cc_zonal_mean = dsERA.cc_zonal_mean
    cc_zonal_mean = cc_zonal_mean[:, 4:-1, :] * 100.
    u_zonal_mean = dsERA.u_zonal_mean
    dsERA.close()

    varname = [tcc, lcc, mcc, hcc, tclwp_per_tcc, stability, windSpeed, mslp, temp_zonal_mean, cc_zonal_mean, u_zonal_mean, shflx, lhflx, sfc_lw_net, sfc_sw_net, sfc_lw_net_clr, sfc_sw_net_clr]
    varnameStr = ['tcc', 'lcc', 'mcc', 'hcc', 'tclwp_per_tcc', 'stability', 'windSpeed', 'mslp', 'temp_zonal_mean', 'cc_zonal_mean', 'u_zonal_mean', 'shflx', 'lhflx', 'sfc_lw_net', 'sfc_sw_net', 'sfc_lw_net_clr', 'sfc_sw_net_clr']

    return tcc, lcc, mcc, hcc, tclwp_per_tcc, stability, windSpeed, mslp, temp_zonal_mean, cc_zonal_mean, u_zonal_mean, shflx, lhflx, sfc_lw_net, sfc_sw_net, sfc_lw_net_clr, sfc_sw_net_clr, latitude, longitude

def calculate_ohu(working_dir):
    tcc, lcc, mcc, hcc, tclwp_per_tcc, stability, windSpeed, mslp, temp_zonal_mean, cc_zonal_mean, u_zonal_mean, shflx, lhflx, sfc_lw_net, sfc_sw_net, sfc_lw_net_clr, sfc_sw_net_clr, latitude, longitude = read_reanalysis_data(working_dir)

    # latitude weighting ----------------------------------
    deg2rad = np.pi / 180.
    coslat = np.asarray(np.cos(deg2rad * latitude))

    # create land mask for heat uptake and latitude weights -----------------
    lon2, lat2 = np.meshgrid(longitude, latitude)
    lat_weights = np.cos(np.deg2rad(lat2))
    is_on_land = landMask(latitude, longitude)
    lat_weights_masked = np.ma.masked_array(lat_weights, mask=is_on_land)

    latmax = -44.75
    latmin = -65
    latmin_ind = int(np.abs(latmin-latitude).argmin())
    latmax_ind = int(np.abs(latmax-latitude).argmin())

    # calculate OHU -----------------------
    netOHU_unmasked = sfc_sw_net + sfc_lw_net - (-1)*shflx - (-1)*lhflx

    # use land mask to only get ocean grid points --------------
    is_on_land = landMask(latitude, longitude)
    netOHU = netOHU_unmasked.where(is_on_land == False)
    southOceanOHU = netOHU[:, latmin_ind:latmax_ind, :]

    netOHU_timeseries = netOHU.groupby('time.year').mean(skipna=True)
    southOceanOHU_timeseries = netOHU_timeseries[:, latmin_ind:latmax_ind, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        globalOHU_annMean1 = np.nansum(np.nanmean(netOHU_timeseries, axis=2) * coslat, axis=1) / np.nansum(np.nanmean(lat_weights_masked, axis=1), axis=0)
        globalOHU_annMean = signal.detrend(globalOHU_annMean1, axis=0)

        southOceanOHU_annMean1 = np.nansum(np.nanmean(southOceanOHU_timeseries, axis=2) * coslat[latmin_ind:latmax_ind],
            axis=1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind, :], axis=1), axis=0)
        southOceanOHU_annMean = signal.detrend(southOceanOHU_annMean1, axis=0)

        globalOHU_2std = np.mean(globalOHU_annMean) + 2. * np.std(globalOHU_annMean)
        southOceanOHU_2std = np.mean(southOceanOHU_annMean) + 2. * np.std(southOceanOHU_annMean)

    # cloud timeseries ---------------
    tcc_masked = tcc.where(is_on_land == False)
    lcc_masked = lcc.where(is_on_land == False)
    tcc_masked = tcc_masked[:, latmin_ind:latmax_ind, :]
    lcc_masked = lcc_masked[:, latmin_ind:latmax_ind, :]
    tcc_timeseries = tcc_masked.groupby('time.year').mean(skipna=True)
    lcc_timeseries = lcc_masked.groupby('time.year').mean(skipna=True)

    # seasonal averages ------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        winterOHU = netOHU.where(netOHU['time.season'] == 'JJA').groupby('time.year').mean(skipna=True)
        springOHU = netOHU.where(netOHU['time.season'] == 'SON').groupby('time.year').mean(skipna=True)
        summerOHU = netOHU.where(netOHU['time.season'] == 'DJF').groupby('time.year').mean(skipna=True)
        fallOHU = netOHU.where(netOHU['time.season'] == 'MAM').groupby('time.year').mean(skipna=True)

        winterTCC = tcc_masked.where(netOHU['time.season'] == 'JJA').groupby('time.year').mean(skipna=True)
        springTCC = tcc_masked.where(netOHU['time.season'] == 'SON').groupby('time.year').mean(skipna=True)
        summerTCC = tcc_masked.where(netOHU['time.season'] == 'DJF').groupby('time.year').mean(skipna=True)
        fallTCC = tcc_masked.where(netOHU['time.season'] == 'MAM').groupby('time.year').mean(skipna=True)

        winterLCC = lcc_masked.where(netOHU['time.season'] == 'JJA').groupby('time.year').mean(skipna=True)
        springLCC = lcc_masked.where(netOHU['time.season'] == 'SON').groupby('time.year').mean(skipna=True)
        summerLCC = lcc_masked.where(netOHU['time.season'] == 'DJF').groupby('time.year').mean(skipna=True)
        fallLCC = lcc_masked.where(netOHU['time.season'] == 'MAM').groupby('time.year').mean(skipna=True)

    southOceanSummerOHU = summerOHU[:, latmin_ind:latmax_ind, :]
    southOceanWinterOHU = winterOHU[:, latmin_ind:latmax_ind, :]
    southOceanFallOHU = fallOHU[:, latmin_ind:latmax_ind, :]
    southOceanSpringOHU = springOHU[:, latmin_ind:latmax_ind, :]

    # timeseries ---------------------------------
    summerOHU_timeseries = signal.detrend((np.sum(np.mean(summerOHU, axis=2) * coslat, axis=1) / np.nansum(np.nanmean(lat_weights_masked, axis=1), axis=0)), axis=0)
    winterOHU_timeseries = signal.detrend((np.sum(np.mean(winterOHU, axis=2) * coslat, axis=1) / np.nansum(np.nanmean(lat_weights_masked, axis=1), axis=0)), axis=0)
    fallOHU_timeseries = signal.detrend((np.sum(np.mean(fallOHU, axis=2) * coslat, axis=1) / np.nansum(np.nanmean(lat_weights_masked, axis=1), axis=0)), axis=0)
    springOHU_timeseries = signal.detrend((np.sum(np.mean(springOHU, axis=2) * coslat, axis=1) / np.nansum(np.nanmean(lat_weights_masked, axis=1), axis=0)), axis=0)

    southOceanSummerOHU_timeseries = signal.detrend((np.sum(np.mean(southOceanSummerOHU, axis = 2) * coslat[latmin_ind:latmax_ind],
    axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)
    southOceanWinterOHU_timeseries = signal.detrend((np.sum(np.mean(southOceanWinterOHU, axis = 2) * coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)
    southOceanFallOHU_timeseries = signal.detrend((np.sum(np.mean(southOceanFallOHU, axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)
    southOceanSpringOHU_timeseries = signal.detrend((np.sum(np.mean(southOceanSpringOHU, axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)

    tccSummer_timeseries = signal.detrend((np.sum(np.mean(summerTCC,axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)
    tccWinter_timeseries = signal.detrend((np.sum(np.mean(winterTCC, axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)
    tccFall_timeseries = signal.detrend((np.sum(np.mean(fallTCC,  axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)
    tccSpring_timeseries = signal.detrend((np.sum(np.mean(springTCC, axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)

    lccSummer_timeseries = signal.detrend((np.sum(np.mean(summerLCC,axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)
    lccWinter_timeseries = signal.detrend((np.sum(np.mean(winterLCC, axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)
    lccFall_timeseries = signal.detrend((np.sum(np.mean(fallLCC,  axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)
    lccSpring_timeseries = signal.detrend((np.sum(np.mean(springLCC, axis = 2)*coslat[latmin_ind:latmax_ind],
        axis = 1) / np.nansum(np.nanmean(lat_weights_masked[latmin_ind:latmax_ind,:], axis = 1), axis = 0)), axis=0)

    # seasonal standard deviations --------------------
    summerOHU_2std = np.nanmean(summerOHU_timeseries) + 2. * (summerOHU_timeseries.std())
    winterOHU_2std = np.nanmean(winterOHU_timeseries) + 2. * (winterOHU_timeseries.std())
    fallOHU_2std = np.nanmean(fallOHU_timeseries) + 2. * (fallOHU_timeseries.std())
    springOHU_2std = np.nanmean(springOHU_timeseries) + 2. * (springOHU_timeseries.std())

    southOceanSummerOHU_2std = np.nanmean(southOceanSummerOHU_timeseries) + 2. * (southOceanSummerOHU_timeseries.std())
    southOceanWinterOHU_2std = np.nanmean(southOceanWinterOHU_timeseries) + 2. * (southOceanWinterOHU_timeseries.std())
    southOceanFallOHU_2std = np.nanmean(southOceanFallOHU_timeseries) + 2. * (southOceanFallOHU_timeseries.std())
    southOceanSpringOHU_2std = np.nanmean(southOceanSpringOHU_timeseries) + 2. * (southOceanSpringOHU_timeseries.std())

    tccSummer_2std = np.nanmean(tccSummer_timeseries) + 2. * (tccSummer_timeseries.std())
    tccWinter_2std = np.nanmean(tccWinter_timeseries) + 2. * (tccWinter_timeseries.std())
    tccFall_2std = np.nanmean(tccFall_timeseries) + 2. * (tccFall_timeseries.std())
    tccSpring_2std = np.nanmean(tccSpring_timeseries) + 2. * (tccSpring_timeseries.std())

    lccSummer_2std = np.nanmean(lccSummer_timeseries) + 2. * (lccSummer_timeseries.std())
    lccWinter_2std = np.nanmean(lccWinter_timeseries) + 2. * (lccWinter_timeseries.std())
    lccFall_2std = np.nanmean(lccFall_timeseries) + 2. * (lccFall_timeseries.std())
    lccSpring_2std = np.nanmean(lccSpring_timeseries) + 2. * (lccSpring_timeseries.std())

    thflx = shflx + lhflx
    shflx = (-1.) * shflx
    lhflx = (-1.) * lhflx
    thflx = (-1.) * thflx
    sfc_total_net = sfc_sw_net + sfc_lw_net
    sfc_total_net_clr = sfc_sw_net_clr + sfc_lw_net_clr
    seasonStr = ['DJF', 'JJA', 'MAM', 'SON']

    return

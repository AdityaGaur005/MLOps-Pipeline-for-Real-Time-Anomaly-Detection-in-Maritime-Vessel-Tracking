import numpy as np
import pandas as pd
from dateutil import tz
import geopandas as gpd
import plotly.express as px
from matplotlib import pyplot as plt
from datetime import timedelta, datetime

def load_hawaii_incidents():
    '''
    Load the primary hawaii incident list and return it as a pandas dataframe
    '''
    # Load the incident index
    incidents = pd.read_csv('incident_data/hawaii_primary_trajectories_of_interest_2017_2020.csv')
    incidents['ais_incident_end_bound_hst'] = pd.to_datetime(incidents['ais_incident_end_bound_hst'])
    incidents['ais_incident_start_bound_hst'] = pd.to_datetime(incidents['ais_incident_start_bound_hst'])
    incidents['report_incident_time_of_interest_hst'] = pd.to_datetime(incidents['report_incident_time_of_interest_hst'])
    incidents['incident_num'] = incidents['incident_num'].astype(int)
    incidents.set_index('incident_num', inplace=True)
    return(incidents)

def load_single_month(year, month):
    '''
    Load a single month of HawaiiCoast_GT AIS data
    '''
    assert year in np.arange(2017, 2021), "Available years are 2017, 2018, 2019, 2020"
    assert month in np.arange(1, 13), f"Available months 1-12, {month} is not a valid month"
    Haw_month = pd.read_csv(f'AIS_data/Hawaii_{year}_{str(month).zfill(2)}.csv').set_index('MMSI')
    Haw_month['datetime_utc'] = pd.to_datetime(Haw_month['datetime_utc'])
    Haw_month['datetime_hst'] = pd.to_datetime(Haw_month['datetime_hst'])
    return(Haw_month)

def load_all_hawaiicoast_ais():
    '''
    Load all the HawaiiCoast_GT AIS files as a dictionary with year, month keys
    and pandas dataframe values.
    '''
    Haw_data = {(i,j):load_single_month(i, j) for i in range(2017, 2021) for j in range(1, 13)}
    return(Haw_data)

def mapbox_plot(plot_points_df, zoom=10, save=False, savename='figure'):
    '''
    Inputs:
    ----------------------------------------------------------------------------
    plot_points_df: pandas dataframe containing the AIS points to plot.
    zoom: optional int between 0 and 20, determines the initial camera zoom of
        the plotly plot. Default 10.
    save: optional bool, determines whether the plot will be saved or not. Default False.
    savename: optional str, if save == True then the plot will be saved as <savename>.html
    '''
    fig = px.scatter_mapbox(plot_points_df, lat='lat', lon='lon', zoom=zoom, mapbox_style='open-street-map', color='vessel_name', hover_data={'datetime_hst':True, 'lat':False, 'lon':False, 'comput_speed_knots':True})
    if save:
        fig.write_html(f'{savename}.html')
    fig.show()

def speed_plot_incident(plot_points_df, incident_time=False, primary_dt = 'datetime_hst', save=False, savename='fig'):
    '''
    Inputs:
    ----------------------------------------------------------------------------
    plot_points_df: pandas dataframe containing the AIS points to plot.
    incident_time: optional datetime object (default False). If provided, this
        plots a vline at the incident time on the speed plot.
    primary_dt: the column header name for the datetime axis of the speed profile.
        Default 'datetime_hst' is tailored to the HawaiiCoast_GT dataset.
    save: optional bool, determines whether the plot will be saved or not. Default False.
    savename: optional str, if save==True then the plot will be saved as <savename>_speed.png
    '''
    fig = plt.figure(figsize=(30,3))
    for mmsi in plot_points_df.index.unique():
        plt.plot(plot_points_df.loc[mmsi, primary_dt], plot_points_df.loc[mmsi,'comput_speed_knots'], label=plot_points_df.loc[mmsi, 'vessel_name'].unique()[0])
    if incident_time:
        plt.vlines(incident_time, 0, plot_points_df['comput_speed_knots'].max(), color='k', linestyle='--')
    plt.xlabel('Date time (HST)')
    plt.ylabel('Computed speed (knots)')
    plt.legend(loc='center left', bbox_to_anchor=(1, .5))
    if save:
        plt.savefig(f'{savename}_speed.png', dpi=1200, bbox_inches='tight')
    plt.show()

def plot_all_traj(data_dict, day1_str, day2_str, primary_dt='datetime_hst', save=False, savename='all_traj'):
    '''
    Plots all AIS points given in data_dict given between day1_str and day2_str.
    Inputs:
    ----------------------------------------------------------------------------
    data_dict: dictionary where data_dict[year, month] = pandas dataframe
        containing the AIS points corresponding to year, month (note this could
        be limited by type, etc as desired.).
    day1_str: string for the starting datetime to be plotted. Format must be the
        same as day2_str.
    day2_str: string for the ending datetime of points to be plotted. Format must
        be the same as day1_str.
    primary_dt: the column header name for the datetime axis of the speed profile.
        Default 'datetime_hst' is tailored to the HawaiiCoast_GT dataset.
    save: optional bool, determines whether the plot will be saved or not. Default False.
    savename: optional str, if save==True then the plot will be saved as <savename>.png
    '''
    days = pd.date_range(day1_str, day2_str, freq='D')
    date1 = days[0]
    date2 = days[-1]
    vessel_df_list = []
    if (date1.month == date2.month) and (date1.year == date2.year):
        year = date1.year
        month = date1.month
        vessel_df_list.append(data_dict[year, month])
    else:
        for fullmonth in pd.date_range(date1, date2, freq='M'):
            year = fullmonth.year
            month = fullmonth.month
            vessel_df_list.append(data_dict[year, month])
    Vessel_df = pd.concat(vessel_df_list)
    plot_points_df = Vessel_df[(Vessel_df[primary_dt] >= date1) & (Vessel_df[primary_dt] <= date2)].sort_values(by=primary_dt)
    mapbox_plot(plot_points_df, save=save, savename=savename)
    return(plot_points_df)

def plot_trajectories(data_dict, mmsi_vals, date1, date2, inc_date = None, primary_dt='datetime_hst', save=False, savename='figure'):
    '''
    Plots the AIS points associated the vessels given in mmsi_vals between
    day1_str and day2_str both visually and as a corresponding set of speed profiles.
    Inputs:
    ----------------------------------------------------------------------------
    data_dict: dictionary where data_dict[year, month] = pandas dataframe
        containing the AIS points corresponding to year, month (note this could
        be limited by type, etc as desired.).
    mmsi_vals: list of mmsi_vals (9 digit integers) corresponding to the vessels
        to plot.
    date1: string for the starting datetime to be plotted. Format must be the
        same as day2_str.
    date2: string for the ending datetime of points to be plotted. Format must
        be the same as day1_str.
    primary_dt: the column header name for the datetime axis of the speed profile.
        Default 'datetime_hst' is tailored to the HawaiiCoast_GT dataset.
    save: optional bool, determines whether the plot will be saved or not. Default False.
    savename: optional str, if save==True then the plot will be saved as <savename>.png
    '''
    vessel_df_list = []
    dr = pd.date_range(date1, date2, freq='D')
    df_key_list = list(set(list((zip(dr.year, dr.month)))))

    for year, month in df_key_list:
        try:
            for mmsi in mmsi_vals:
                vessel_df_list.append(data_dict[year, month].loc[mmsi])
        except:
            vessel_df_list.append(data_dict[year, month].loc[mmsi_vals])
    '''
    if (date1.month == date2.month) and (date1.year == date2.year):
        year = date1.year
        month = date1.month
        try:
            for mmsi in mmsi_vals:
                vessel_df_list.append(data_dict[year, month].loc[mmsi])
        except:
            vessel_df_list.append(data_dict[year, month].loc[mmsi_vals])

    else:
        for fullmonth in pd.date_range(date1, date2, freq='M'):

            year = fullmonth.year
            month = fullmonth.month
            if type(mmsi_vals) is list:
                for mmsi in mmsi_vals:
                    vessel_df_list.append(data_dict[year, month].loc[mmsi])
            else:
                vessel_df_list.append(data_dict[year, month].loc[mmsi])
    '''
    Vessel_df = pd.concat(vessel_df_list)

    plot_points_df = Vessel_df[(Vessel_df[primary_dt] >= date1) & (Vessel_df[primary_dt] <= date2)].sort_values(by=primary_dt)
    mapbox_plot(plot_points_df, save=save, savename=savename)

    if inc_date is not None:
        speed_plot_incident(plot_points_df, inc_date, primary_dt=primary_dt, save=save, savename=savename)
    else:
        speed_plot_incident(plot_points_df, primary_dt=primary_dt, save=save, savename=savename)
    return(plot_points_df)

import streamlit as st
import ee
import geemap.foliumap as geemap
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


# Initialize the Earth Engine module.
ee.Initialize()

# Function to get basins and sub-basins.
def get_basins():
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    basins_list = dataset.aggregate_array("Basin").sort().distinct().getInfo()
    return basins_list
def get_sub_basins(selected_basin):
    sub_basins_list = ['None']
    if selected_basin and selected_basin != 'None':
        dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
        selected_basin_feature = dataset.filter(ee.Filter.eq('Basin', selected_basin))
        sub_basins_list += selected_basin_feature.aggregate_array('Sub_Basin').sort().distinct().getInfo()
    return sub_basins_list

# Function to get NDVI image clipped by the sub-basin and dates
def get_ndvi_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(0.0001).copyProperties(img,['system:time_start','date','system:time_end'])

    # Filter the MODIS NDVI collection by date
    ndvi_collection = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start_date, end_date).select('NDVI').map(scale_index)

    # Calculate the mean NDVI image
    mean_ndvi_image = ndvi_collection.mean().clip(sub_basin_feature)

    return mean_ndvi_image
def create_ndvi_timeseries(selected_sub_basin, from_date, to_date):
    
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(0.0001).copyProperties(img,['system:time_start','date','system:time_end'])

    # Filter the MODIS NDVI collection by date
    ndvi_collection = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start_date, end_date).select('NDVI').map(scale_index)

    # Create a list of dates and mean NDVI values
    timeseries = ndvi_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'NDVI': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=250
        ).get('NDVI')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'NDVI']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'NDVI'])
    # df['date'] = pd.to_datetime(df['date'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['ndvi_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6)) #10 x 6
    df.plot(x='date', y='NDVI', ax=ax, legend=True, title='NDVI Time Series')
    plt.xlabel('Date',fontsize=6)
    plt.ylabel('Mean NDVI')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_era_temp_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.subtract(273).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    temp_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_date, end_date).select('temperature_2m').map(scale_index)

    # Calculate the mean temperature image
    mean_temp_image = temp_collection.mean().clip(sub_basin_feature)

    return mean_temp_image
def create_temp_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.subtract(273).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    temp_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_date, end_date).select('temperature_2m').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = temp_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'temperature': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=1000  # Adjust scale according to your data resolution
        ).get('temperature_2m')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'temperature']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'temperature'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['temp_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='temperature', ax=ax, legend=True, title='Temperature Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Mean Temperature')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_evi_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(0.0001).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    evi_collection = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start_date, end_date).select('EVI').map(scale_index)

    # Calculate the mean temperature image
    mean_evi_image = evi_collection.mean().clip(sub_basin_feature)

    return mean_evi_image
def create_evi_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(0.0001).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    evi_collection = ee.ImageCollection("MODIS/061/MOD13Q1").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('EVI').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = evi_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'evi': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=250  # Adjust scale according to your data resolution
        ).get('EVI')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'evi']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'evi'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['evi_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='evi', ax=ax, legend=True, title='EVI Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Mean EVI')
    plt.grid(True)
    plt.tight_layout()

    return fig


# def get_et_image(selected_sub_basin, from_date, to_date):
#     if selected_sub_basin == 'None':
#         return None
#     dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
#     sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

#     # Convert dates to ee.Date objects
#     start_date = ee.Date(from_date)
#     end_date = ee.Date(to_date)

#     def scale_index(img):
#         return img.multiply(0.1).copyProperties(img,['system:time_start','date','system:time_end'])
#     # Filter the ERA temperature collection by date
#     et_collection = ee.ImageCollection("MODIS/061/MOD16A2").filterDate(start_date, end_date).select('ET').map(scale_index)

#     # Calculate the mean temperature image
#     mean_et_image = et_collection.mean().clip(sub_basin_feature)

#     return mean_et_image
# def create_et_timeseries(selected_sub_basin, from_date, to_date):
#     if selected_sub_basin == 'None':
#         return None
#     dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
#     sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

#     # Convert dates to ee.Date objects
#     start_date = ee.Date(from_date)
#     end_date = ee.Date(to_date)
#     def scale_index(img):
#         return img.multiply(0.1).copyProperties(img,['system:time_start','date','system:time_end'])
#     # Filter the ERA temperature collection by date
#     et_collection = ee.ImageCollection("MODIS/061/MOD16A2").filterDate(start_date, end_date).select('ET').map(scale_index)

#     # Create a list of dates and mean temperature values
#     timeseries = et_collection.map(lambda image: ee.Feature(None, {
#         'date': image.date().format(),
#         'ET': image.reduceRegion(
#             reducer=ee.Reducer.mean(),
#             geometry=sub_basin_feature,
#             scale=500  # Adjust scale according to your data resolution
#         ).get('ET')
#     }))

#     # Convert to a Pandas DataFrame
#     timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'ET']).values().get(0).getInfo()
#     df = pd.DataFrame(timeseries_list, columns=['date', 'ET'])
#     df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
#     st.session_state['et_chart_data'] = df

#     # Create a time-series plot
#     fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
#     df.plot(x='date', y='ET', ax=ax, legend=True, title='Evapotranspiration Time Series')
#     plt.xlabel('Date', fontsize=6)
#     plt.ylabel('Mean ET')
#     plt.grid(True)
#     plt.tight_layout()

#     return fig

# def get_kbdi_image(selected_sub_basin, from_date, to_date):
#     if selected_sub_basin == 'None':
#         return None
#     dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
#     sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

#     # Convert dates to ee.Date objects
#     start_date = ee.Date(from_date)
#     end_date = ee.Date(to_date)

#     def scale_index(img):
#         return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
#     # Filter the ERA temperature collection by date
#     kbdi_collection = ee.ImageCollection("UTOKYO/WTLAB/KBDI/v1").filterDate(start_date, end_date).select('KBDI').map(scale_index)

#     # Calculate the mean temperature image
#     mean_kbdi_image = kbdi_collection.mean().clip(sub_basin_feature)

#     return mean_kbdi_image
# def create_kbdi_timeseries(selected_sub_basin, from_date, to_date):
#     if selected_sub_basin == 'None':
#         return None
#     dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
#     sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

#     # Convert dates to ee.Date objects
#     start_date = ee.Date(from_date)
#     end_date = ee.Date(to_date)
#     def scale_index(img):
#         return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
#     # Filter the ERA temperature collection by date
#     kbdi_collection = ee.ImageCollection("UTOKYO/WTLAB/KBDI/v1").filterDate(start_date, end_date).select('KBDI').map(scale_index)

#     # Create a list of dates and mean temperature values
#     timeseries = kbdi_collection.map(lambda image: ee.Feature(None, {
#         'date': image.date().format(),
#         'KBDI': image.reduceRegion(
#             reducer=ee.Reducer.mean(),
#             geometry=sub_basin_feature,
#             scale=4000  # Adjust scale according to your data resolution
#         ).get('KBDI')
#     }))

#     # Convert to a Pandas DataFrame
#     timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'KBDI']).values().get(0).getInfo()
#     df = pd.DataFrame(timeseries_list, columns=['date', 'KBDI'])
#     df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
#     st.session_state['kbdi_chart_data'] = df

#     # Create a time-series plot
#     fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
#     df.plot(x='date', y='KBDI', ax=ax, legend=True, title='KBDI Time Series')
#     plt.xlabel('Date', fontsize=6)
#     plt.ylabel('Mean KBDI')
#     plt.grid(True)
#     plt.tight_layout()

#     return fig

# def get_lst_image(selected_sub_basin, from_date, to_date):
#     if selected_sub_basin == 'None':
#         return None
#     dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
#     sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

#     # Convert dates to ee.Date objects
#     start_date = ee.Date(from_date)
#     end_date = ee.Date(to_date)

#     def scale_index(img):
#         return img.multiply(0.02).subtract(273).copyProperties(img,['system:time_start','date','system:time_end'])
#     # Filter the ERA temperature collection by date
#     lst_collection = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(start_date, end_date).select('LST_Day_1km').map(scale_index)

#     # Calculate the mean temperature image
#     mean_lst_image = lst_collection.mean().clip(sub_basin_feature)

#     return mean_lst_image
# def create_lst_timeseries(selected_sub_basin, from_date, to_date):
#     if selected_sub_basin == 'None':
#         return None
#     dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
#     sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

#     # Convert dates to ee.Date objects
#     start_date = ee.Date(from_date)
#     end_date = ee.Date(to_date)
#     def scale_index(img):
#         return img.multiply(0.02).subtract(273).copyProperties(img,['system:time_start','date','system:time_end'])
#     # Filter the ERA temperature collection by date
#     lst_collection = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(start_date, end_date).select('LST_Day_1km').map(scale_index)

#     # Create a list of dates and mean temperature values
#     timeseries = lst_collection.map(lambda image: ee.Feature(None, {
#         'date': image.date().format(),
#         'LST': image.reduceRegion(
#             reducer=ee.Reducer.mean(),
#             geometry=sub_basin_feature,
#             scale=1000  # Adjust scale according to your data resolution
#         ).get('LST_Day_1km')
#     }))

#     # Convert to a Pandas DataFrame
#     timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'LST']).values().get(0).getInfo()
#     df = pd.DataFrame(timeseries_list, columns=['date', 'LST'])
#     df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
#     st.session_state['lst_chart_data'] = df

#     # Create a time-series plot
#     fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
#     df.plot(x='date', y='LST', ax=ax, legend=True, title='LST Time Series')
#     plt.xlabel('Date', fontsize=6)
#     plt.ylabel('Mean LST')
#     plt.grid(True)
#     plt.tight_layout()

#     return fig
# def get_landcover_image(selected_sub_basin):
#     if selected_sub_basin == 'None':
#         return None
#     dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
#     sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))
#     landcover_data = ee.ImageCollection('ESA/WorldCover/v100').filterBounds(sub_basin_feature).filterDate('2018-01-01','2023-01-01').first().clip(sub_basin_feature)
#     return landcover_data

################################
################################
################################
# def get_srtm_image(selected_sub_basin):
#     if selected_sub_basin == 'None':
#         return None
#     dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
#     sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))
#     srtm_data = ee.Image("CGIAR/SRTM90_V4").clip(sub_basin_feature)
#     return srtm_data

def get_landcover_image(selected_sub_basin):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = '2020-01-01'
    end_date = '2020-01-02'

    # Filter the ERA temperature collection by date
    landCover_collection = ee.ImageCollection('ESA/WorldCover/v100').filterDate(start_date, end_date)

    # Calculate the mean temperature image
    mean_landcover_image = landCover_collection.first().clip(sub_basin_feature)

    return mean_landcover_image

def get_lst_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(0.02).subtract(273).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    lst_collection = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(start_date, end_date).select('LST_Day_1km').map(scale_index)

    # Calculate the mean temperature image
    mean_lst_image = lst_collection.mean().clip(sub_basin_feature)

    return mean_lst_image
def create_lst_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(0.02).subtract(273).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    lst_collection = ee.ImageCollection("MODIS/061/MOD11A2").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('LST_Day_1km').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = lst_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'LST_Day_1km': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=1000  # Adjust scale according to your data resolution
        ).get('LST_Day_1km')
    }))
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'LST_Day_1km']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'LST_Day_1km'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['lst_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='LST_Day_1km', ax=ax, legend=True, title='LST (C) Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Mean LST')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_srtm_image(selected_sub_basin):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))
    srtm_data = ee.Image("CGIAR/SRTM90_V4").clip(sub_basin_feature)
    return srtm_data

def get_population_image(selected_sub_basin):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))
    population_data = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex_cons_unadj").select('population').filterBounds(sub_basin_feature).filterDate('2020-01-01','2023-12-31').mean().clip(sub_basin_feature)
    print(population_data.getInfo())
    return population_data

def get_kbdi_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    kbdi_collection = ee.ImageCollection("UTOKYO/WTLAB/KBDI/v1").filterDate(start_date, end_date).select('KBDI').map(scale_index)

    # Calculate the mean temperature image
    mean_kbdi_image = kbdi_collection.mean().clip(sub_basin_feature)

    return mean_kbdi_image
def create_kbdi_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    kbdi_collection = ee.ImageCollection("UTOKYO/WTLAB/KBDI/v1").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('KBDI').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = kbdi_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'KBDI': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=4000  # Adjust scale according to your data resolution
        ).get('KBDI')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'KBDI']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'KBDI'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['kbdi_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='KBDI', ax=ax, legend=True, title='KBDI Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Mean KBDI')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_et_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(0.1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    et_collection = ee.ImageCollection("MODIS/061/MOD16A2").filterDate(start_date, end_date).select('ET').map(scale_index)

    # Calculate the mean temperature image
    mean_et_image = et_collection.mean().clip(sub_basin_feature)

    return mean_et_image
def create_et_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(0.1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    et_collection = ee.ImageCollection("MODIS/061/MOD16A2").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('ET').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = et_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'ET': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=1000  # Adjust scale according to your data resolution
        ).get('ET')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'ET']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'ET'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['et_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='ET', ax=ax, legend=True, title='ET Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Mean Evapotranspiration (ET)')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_lhf_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(10000).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    lhf_collection = ee.ImageCollection("MODIS/061/MOD16A2").filterDate(start_date, end_date).select('LE').map(scale_index)

    # Calculate the mean temperature image
    mean_lhf_image = lhf_collection.mean().clip(sub_basin_feature)

    return mean_lhf_image
def create_lhf_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataslhf = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataslhf.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(10000).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    lhf_collection = ee.ImageCollection("MODIS/061/MOD16A2").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('LE').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = lhf_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'LE': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=1000  # Adjust scale according to your data resolution
        ).get('LE')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'LE']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'LE'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['lhf_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='LE', ax=ax, legend=True, title='LE Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Mean Latent Heat Flux (LE)')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_pet_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(0.1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    pet_collection = ee.ImageCollection("MODIS/061/MOD16A2").filterDate(start_date, end_date).select('PET').map(scale_index)

    # Calculate the mean temperature image
    mean_pet_image = pet_collection.mean().clip(sub_basin_feature)

    return mean_pet_image
def create_pet_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataspet = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataspet.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(0.1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    pet_collection = ee.ImageCollection("MODIS/061/MOD16A2").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('PET').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = pet_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'PET': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=1000  # Adjust scale according to your data resolution
        ).get('PET')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'PET']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'PET'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['pet_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='PET', ax=ax, legend=True, title='PET Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Potential Evapotranspiration (PET)')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_LAI_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(0.1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    LAI_collection = ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(start_date, end_date).select('Lai_500m').map(scale_index)

    # Calculate the mean temperature image
    mean_LAI_image = LAI_collection.mean().clip(sub_basin_feature)

    return mean_LAI_image
def create_LAI_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(0.1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    LAI_collection = ee.ImageCollection("MODIS/061/MOD15A2H").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('Lai_500m').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = LAI_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'Lai_500m': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=500  # Adjust scale according to your data resolution
        ).get('Lai_500m')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'Lai_500m']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'Lai_500m'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['LAI_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='Lai_500m', ax=ax, legend=True, title='LAI Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Leaf Area Index (LAI)')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_Precp_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    Precp_collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start_date, end_date).select('precipitation').map(scale_index)

    # Calculate the mean temperature image
    mean_Precp_image = Precp_collection.mean().clip(sub_basin_feature)

    return mean_Precp_image
def create_Precp_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    Precp_collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('precipitation').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = Precp_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'precipitation': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=5566  # Adjust scale according to your data resolution
        ).get('precipitation')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'precipitation']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'precipitation'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['Precp_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='precipitation', ax=ax, legend=True, title='Precipitation Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Precipitation mm/day')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_snow_cover_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    snow_cover_collection = ee.ImageCollection("MODIS/061/MOD10A1").filterDate(start_date, end_date).select('NDSI_Snow_Cover').map(scale_index)

    # Calculate the mean temperature image
    mean_snow_cover_image = snow_cover_collection.mean().clip(sub_basin_feature)

    return mean_snow_cover_image
def create_snow_cover_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(0.01).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    snow_cover_collection = ee.ImageCollection("MODIS/061/MOD10A1").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('NDSI_Snow_Cover').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = snow_cover_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'NDSI_Snow_Cover': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=500  # Adjust scale according to your data resolution
        ).get('NDSI_Snow_Cover')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'NDSI_Snow_Cover']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'NDSI_Snow_Cover'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['snow_cover_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='NDSI_Snow_Cover', ax=ax, legend=True, title='NDSI (Snow Cover) Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('NDSI')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_soil_moisture_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    soil_moisture_collection = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture").filterDate(start_date, end_date).select('ssm').map(scale_index)

    # Calculate the mean temperature image
    mean_soil_moisture_image = soil_moisture_collection.mean().clip(sub_basin_feature)

    return mean_soil_moisture_image
def create_soil_moisture_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    soil_moisture_collection = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('ssm').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = soil_moisture_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'ssm': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=10000  # Adjust scale according to your data resolution
        ).get('ssm')
    }))
    try:
    # Convert to a Pandas DataFrame
        timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'ssm']).values().get(0).getInfo()
        df = pd.DataFrame(timeseries_list, columns=['date', 'ssm'])
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
        st.session_state['soil_moisture_chart_data'] = df

        # Create a time-series plot
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
        df.plot(x='date', y='ssm', ax=ax, legend=True, title='Soil Moisture Time Series')
        plt.xlabel('Date', fontsize=6)
        plt.ylabel('Soil Moisture mm')
        plt.grid(True)
        plt.tight_layout()

        return fig
    except:
        st.write('No Data Available')

def get_soil_temperature_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(1).subtract(273).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    soil_temperature_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_date, end_date).select('soil_temperature_level_1').map(scale_index)

    # Calculate the mean temperature image
    mean_soil_temperature_image = soil_temperature_collection.mean().clip(sub_basin_feature)

    return mean_soil_temperature_image
def create_soil_temperature_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(1).subtract(273).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    soil_temperature_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('soil_temperature_level_1').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = soil_temperature_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'soil_temperature_level_1': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=11132  # Adjust scale according to your data resolution
        ).get('soil_temperature_level_1')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'soil_temperature_level_1']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'soil_temperature_level_1'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['soil_temperature_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='soil_temperature_level_1', ax=ax, legend=True, title='Soil Temperature Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Soil Temperature (C)')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_transpiration_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    transpiration_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_date, end_date).select('evaporation_from_vegetation_transpiration_sum').map(scale_index)

    # Calculate the mean temperature image
    mean_transpiration_image = transpiration_collection.mean().clip(sub_basin_feature)

    return mean_transpiration_image
def create_transpiration_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        return img.multiply(1).copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    transpiration_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select('evaporation_from_vegetation_transpiration_sum').map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = transpiration_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'evaporation_from_vegetation_transpiration_sum': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=11132  # Adjust scale according to your data resolution
        ).get('evaporation_from_vegetation_transpiration_sum')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'evaporation_from_vegetation_transpiration_sum']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'evaporation_from_vegetation_transpiration_sum'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['transpiration_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='evaporation_from_vegetation_transpiration_sum', ax=ax, legend=True, title='Transpiration Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('Transipiration (m)')
    plt.grid(True)
    plt.tight_layout()

    return fig

def get_windspeed_image(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    def scale_index(img):
        ws = img.expression("sqrt((u**2)+(v**2))",{
                "u": img.select('u_component_of_wind_10m'),
                "v": img.select('v_component_of_wind_10m'),
              }).rename('windspeed')
        return ws.copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    windspeed_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_date, end_date).select(['u_component_of_wind_10m','v_component_of_wind_10m']).map(scale_index)

    # Calculate the mean temperature image
    mean_windspeed_image = windspeed_collection.mean().clip(sub_basin_feature)

    return mean_windspeed_image
def create_windspeed_timeseries(selected_sub_basin, from_date, to_date):
    if selected_sub_basin == 'None':
        return None
    dataset = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')
    sub_basin_feature = dataset.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)
    def scale_index(img):
        ws = img.expression("sqrt((u**2)+(v**2))",{
                "u": img.select('u_component_of_wind_10m'),
                "v": img.select('v_component_of_wind_10m'),
              }).rename('windspeed')
        return ws.copyProperties(img,['system:time_start','date','system:time_end'])
    # Filter the ERA temperature collection by date
    windspeed_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterBounds(sub_basin_feature).filterDate(start_date, end_date).select(['u_component_of_wind_10m','v_component_of_wind_10m']).map(scale_index)

    # Create a list of dates and mean temperature values
    timeseries = windspeed_collection.map(lambda image: ee.Feature(None, {
        'date': image.date().format(),
        'windspeed': image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sub_basin_feature,
            scale=11132  # Adjust scale according to your data resolution
        ).get('windspeed')
    }))

    # Convert to a Pandas DataFrame
    timeseries_list = timeseries.reduceColumns(ee.Reducer.toList(2), ['date', 'windspeed']).values().get(0).getInfo()
    df = pd.DataFrame(timeseries_list, columns=['date', 'windspeed'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %Y')
    st.session_state['windspeed_chart_data'] = df

    # Create a time-series plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    df.plot(x='date', y='windspeed', ax=ax, legend=True, title='windspeed Time Series')
    plt.xlabel('Date', fontsize=6)
    plt.ylabel('windspeed (m/s)')
    plt.grid(True)
    plt.tight_layout()

    return fig

###########################################
###########################################


def get_ndvi_image_for_roi(geojson_data, from_date, to_date):
    # Convert the GeoJSON object to an Earth Engine Geometry
    roi = ee.FeatureCollection(geojson_data)

    # Convert dates to ee.Date objects
    start_date = ee.Date(from_date)
    end_date = ee.Date(to_date)

    # Filter the MODIS NDVI collection by date
    ndvi_collection = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start_date, end_date).select('NDVI')

    # Calculate the mean NDVI image
    mean_ndvi_image = ndvi_collection.mean().clip(roi)

    return mean_ndvi_image

def nullify_map():
    st.session_state['ndvi_image'] = None
    st.session_state['srtm_image'] = None
    st.session_state['temp_image'] = None
    st.session_state['lst_image'] = None
    st.session_state['evi_image'] = None
    st.session_state['kbdi_image'] = None
    st.session_state['et_image'] = None
    st.session_state['landcover_image'] = None
    st.session_state['lhf_image'] = None
    st.session_state['pet_image'] = None
    st.session_state['LAI_image'] = None
    st.session_state['Precp_image'] = None
    st.session_state['transpiration_image'] = None
    st.session_state['snow_cover_image'] = None
    st.session_state['population_image'] = None
    st.session_state['soil_moisture_image'] = None
    st.session_state['soil_temperature_image'] = None
    st.session_state['windspeed_image'] = None

# Function to display map
def display_map(selected_index,selected_basin=None, selected_sub_basin=None, ndvi_image=None, srtm_image=None,transpiration_image=None,windspeed_image=None, temp_image=None,soil_temperature_image=None,soil_moisture_image=None,population_image=None, snow_cover_image=None, Precp_image=None, LAI_image=None, landcover_image=None, evi_image=None, et_image=None,pet_image=None, lhf_image=None, kbdi_image=None,lst_image=None, geojson_data=None):
    Map = geemap.Map(add_google_map=False)
    basin_feature_collection = ee.FeatureCollection('projects/ee-mspkafg/assets/1-final_validated_data/SubBasins')

    # Set the map center to a default location first
    Map.setCenter(70, 33, 7)

    # Clear all layers before adding new ones
    Map.layers = []

    # Add GeoJSON data to the map if available
    if geojson_data:
        geojson_layer = ee.FeatureCollection(geojson_data)
        Map.centerObject(geojson_layer)
        Map.addLayer(geojson_layer.style(color='red',fillColor='#FFFFFF00'), {}, "Uploaded GeoJSON")

    # below is select basin - sub basin
    if selected_basin and selected_basin != 'None':
        basin_feature = basin_feature_collection.filter(ee.Filter.eq('Basin', selected_basin))
        Map.addLayer(basin_feature, {}, 'Basin')
        Map.centerObject(basin_feature, 7)

    if selected_sub_basin and selected_sub_basin != 'None':
        sub_basin_feature = basin_feature_collection.filter(ee.Filter.eq('Sub_Basin', selected_sub_basin))
        Map.addLayer(sub_basin_feature.style(**{'color': 'red'}), {}, 'Sub-Basin')
        Map.centerObject(sub_basin_feature)

    if selected_index == 'NDVI' and ndvi_image:
        ndvi_vis_params = {'min': 0, 'max': 1, 'palette': ['0000FF', '00FF00', 'FF0000']}
        Map.addLayer(ndvi_image, ndvi_vis_params, 'NDVI')
        Map.add_colormap(label="NDVI Colorbar", position=(35,5), vmin=ndvi_vis_params['min'], vmax=ndvi_vis_params['max'], vis_params=ndvi_vis_params)
        # Map.add_widget(st.session_state['ndvi_chart'])
    
    if selected_index == 'Land Surface Temperature' and lst_image:
        lst_vis_params = {'min': -10, 'max': 75, 'palette': ['blue','cyan','green','yellow','orange','red']}
        Map.addLayer(lst_image, lst_vis_params, 'LST')
        Map.add_colormap(label="Land Surface Temp. (C)", position=(35,5), vmin=lst_vis_params['min'], vmax=lst_vis_params['max'], vis_params=lst_vis_params)
        # Map.add_widget(st.session_state['ndvi_chart'])
    
    if selected_index == 'Keetch-Byram Drought Index' and kbdi_image:
        kbdi_vis_params = {'min': 0, 'max': 800, 'palette': ['001a4d', '003cb3', '80aaff', '336600', 'cccc00', 'cc9900', 'cc6600','660033']}
        Map.addLayer(kbdi_image, kbdi_vis_params, 'KBDI')
        Map.add_colormap(label="Keetch-Byram Drought Index", position=(35,5), vmin=kbdi_vis_params['min'], vmax=kbdi_vis_params['max'], vis_params=kbdi_vis_params)
        # Map.add_widget(st.session_state['ndvi_chart'])
    
    if selected_index=="SRTM" and srtm_image:
        srtm_vis_params = {'min': 0, 'max': 3000, 'palette': ['0000FF', 'FF0000']}
        Map.addLayer(srtm_image, srtm_vis_params, 'SRTM')
        Map.add_colormap(label="Elevation (mASL)", position=(35,5), vmin=srtm_vis_params['min'], vmax=srtm_vis_params['max'], vis_params=srtm_vis_params)
    
    if selected_index=="Population" and population_image:
        population_vis_params = {'min': 0, 'max': 100, 'palette': ['0000FF', 'FF0000']}
        Map.addLayer(population_image, population_vis_params, 'Population Density')
        Map.add_colormap(label="Population Density (2019)", position=(35,5), vmin=population_vis_params['min'], vmax=population_vis_params['max'], vis_params=population_vis_params)
    
    if selected_index=="Land Cover (2020)" and landcover_image:
        landcover_image = landcover_image.remap([10,20,30,40,50,60,70,80,90,95,100],[0,10,20,30,40,50,60,70,80,90,100])
        lc_vis_params = {'min': 0, 'max': 100, 'palette': ['006400', 'ffbb22','ffff4c','f096ff','fa0000','b4b4b4','f0f0f0','0064c8','0096a0','00cf75','fae6a0']}
        Map.addLayer(landcover_image, lc_vis_params, 'Land Cover 2020')
        #Adding Color-Bar Discrete
        num_classes = 11
        colors = ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000', '#b4b4b4',
                '#f0f0f0', '#0064c8', '#0096a0', '#00cf75', '#fae6a0']
        fig, ax = plt.subplots(figsize=(3, 1))
        fig.subplots_adjust(bottom=0.5)
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        bounds = np.linspace(0, num_classes, num_classes + 1)
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax, orientation='horizontal', ticks=bounds + 0.5)
        cb.set_ticklabels([str(i) for i in range(num_classes + 1)])
        ax.set_xlabel('Land Cover Classes')
        # ax.set_title('Land Cover Classes', fontsize=6, loc='center')
        Map.add_widget(fig,position="bottomright")
        # Map.add_colormap(tick_size=8, discrete=True,label="LC 2020 Colorbar", position=(35,5), vmin=lc_vis_params['min'], vmax=lc_vis_params['max'], vis_params=lc_vis_params)
    #elif was here
    if selected_index == 'Air Temperature' and temp_image:
        temp_vis_params = {'min': -10, 'max': 55, 'palette': ['blue', 'cyan', 'green','yellow','orange','red']}
        Map.addLayer(temp_image, temp_vis_params, 'Air Temperature')
        Map.add_colormap(label="Air Temperature (C)", position=(35,5), vmin=temp_vis_params['min'], vmax=temp_vis_params['max'], vis_params=temp_vis_params)
    if selected_index == 'Soil Moisture' and soil_moisture_image:
        soil_moisture_vis_params = {'min': 0, 'max': 20, 'palette': ['0300ff', '418504', 'efff07', 'efff07', 'ff0303']}
        Map.addLayer(soil_moisture_image, soil_moisture_vis_params, 'Soil Moisture')
        Map.add_colormap(label="Soil Moisture (mm) Mean", position=(35,5), vmin=soil_moisture_vis_params['min'], vmax=soil_moisture_vis_params['max'], vis_params=soil_moisture_vis_params)
    
    if selected_index == 'Soil Temperature' and soil_temperature_image:
        soil_temperature_vis_params = {'min': -5, 'max': 40, 'palette': ['000080', '0000d9', '4000ff', '8000ff', '0080ff', '00ffff','00ff80', '80ff00', 'daff00', 'ffff00', 'fff500', 'ffda00','ffb000', 'ffa400', 'ff4f00', 'ff2500', 'ff0a00', 'ff00ff']}
        Map.addLayer(soil_temperature_image, soil_temperature_vis_params, 'Soil Temperature')
        Map.add_colormap(label="Soil Temp. (C) Mean", position=(35,5), vmin=soil_temperature_vis_params['min'], vmax=soil_temperature_vis_params['max'], vis_params=soil_temperature_vis_params)

    if selected_index == 'Transpiration (TP)' and transpiration_image:
        transpiration_vis_params = {'min': 0, 'max': 5, 'palette': ['000080', '0000d9', '4000ff', '8000ff', '0080ff', '00ffff','00ff80', '80ff00', 'daff00', 'ffff00', 'fff500', 'ffda00','ffb000', 'ffa400', 'ff4f00', 'ff2500', 'ff0a00', 'ff00ff']}
        Map.addLayer(transpiration_image, transpiration_vis_params, 'Transpiration')
        Map.add_colormap(label="Transpiration (m) Mean", position=(35,5), vmin=transpiration_vis_params['min'], vmax=transpiration_vis_params['max'], vis_params=transpiration_vis_params)

    if selected_index == 'Wind Speed' and windspeed_image:
        windspeed_vis_params = {'min': 0, 'max': 6, 'palette': ['000080', '0000d9', '4000ff', '8000ff', '0080ff', '00ffff','00ff80', '80ff00', 'daff00', 'ffff00', 'fff500', 'ffda00','ffb000', 'ffa400', 'ff4f00', 'ff2500', 'ff0a00', 'ff00ff']}
        Map.addLayer(windspeed_image, windspeed_vis_params, 'windspeed')
        Map.add_colormap(label="windspeed (m) Mean", position=(35,5), vmin=windspeed_vis_params['min'], vmax=windspeed_vis_params['max'], vis_params=windspeed_vis_params)


    if selected_index == 'Evapotranspiration ET' and et_image:
        et_vis_params = {'min': 0, 'max': 20, 'palette': ['ffffff', 'fcd163', '99b718', '66a000', '3e8601', '207401', '056201','004c00', '011301']}
        Map.addLayer(et_image, et_vis_params, 'ET')
        Map.add_colormap(label="ET (kg/m^2)", position=(35,5), vmin=et_vis_params['min'], vmax=et_vis_params['max'], vis_params=et_vis_params)
    
    if selected_index == 'Latent Heat Flux (LE)' and lhf_image:
        lhf_vis_params = {'min': 500, 'max': 1671, 'palette': ['ffffff', 'fcd163', '99b718', '66a000', '3e8601', '207401', '056201','004c00', '011301']}
        Map.addLayer(lhf_image, lhf_vis_params, 'Latent Heat Flux')
        Map.add_colormap(label="LE (J/m^2/day)", position=(35,5), vmin=lhf_vis_params['min'], vmax=lhf_vis_params['max'], vis_params=lhf_vis_params)
    
    if selected_index == 'Potential Evapotranspiration (PET)' and pet_image:
        pet_vis_params = {'min': 0, 'max': 75, 'palette': ['ffffff', 'fcd163', '99b718', '66a000', '3e8601', '207401', '056201','004c00', '011301']}
        Map.addLayer(pet_image, pet_vis_params, 'Potential ET')
        Map.add_colormap(label="PET (kg/m^2)", position=(35,5), vmin=pet_vis_params['min'], vmax=pet_vis_params['max'], vis_params=pet_vis_params)
    
    if selected_index == 'Leaf Area Index' and LAI_image:
        LAI_vis_params = {'min': -1, 'max': 1, 'palette': ['ffffff', 'ce7e45', 'df923d', 'f1b555', 'fcd163', '99b718', '74a901', '66a000', '529400', '3e8601', '207401', '056201', '004c00', '023b01',  '012e01', '011d01', '011301']}
        Map.addLayer(LAI_image, LAI_vis_params, 'LAI')
        Map.add_colormap(label="LAI", position=(35,5), vmin=LAI_vis_params['min'], vmax=LAI_vis_params['max'], vis_params=LAI_vis_params)

    if selected_index == 'Precipitation' and Precp_image:
        Precp_vis_params = {'min': 0, 'max': 10, 'palette': ['001137', '0aab1e', 'e7eb05', 'ff4a2d', 'e90000']}
        Map.addLayer(Precp_image, Precp_vis_params, 'Precipitation mm/day')
        Map.add_colormap(label="Mean Precipitation", position=(35,5), vmin=Precp_vis_params['min'], vmax=Precp_vis_params['max'], vis_params=Precp_vis_params)

    if selected_index == 'Snow Cover' and snow_cover_image:
        snow_cover_vis_params = {'min': -1, 'max': 1, 'palette': ['black', '0dffff', '0524ff']}
        Map.addLayer(snow_cover_image, snow_cover_vis_params, 'NDSI')
        Map.add_colormap(label="NDSI (Snow Cover) Mean", position=(35,5), vmin=snow_cover_vis_params['min'], vmax=snow_cover_vis_params['max'], vis_params=snow_cover_vis_params)


    if selected_index == 'Keetch-Byram Drought Index' and kbdi_image:
        kbdi_vis_params = {'min': 0, 'max': 800, 'palette': ['001a4d', '003cb3', '80aaff', '336600', 'cccc00', 'cc9900', 'cc6600','660033']}
        Map.addLayer(kbdi_image, kbdi_vis_params, 'KBDI')
        Map.add_colormap(label="KBDI Colorbar", position=(35,5), vmin=kbdi_vis_params['min'], vmax=kbdi_vis_params['max'], vis_params=kbdi_vis_params)
    #elif here
    if selected_index == 'Enhanced Vegetation Index (EVI)' and evi_image:
        evi_vis_params = {'min': 0, 'max': 1, 'palette': ['ffffff', 'ce7e45', 'df923d', 'f1b555', 'fcd163', '99b718', '74a901','66a000', '529400', '3e8601', '207401', '056201', '004c00', '023b01', '012e01', '011d01', '011301']}
        Map.addLayer(evi_image, evi_vis_params, 'EVI')
        Map.add_colormap(label="EVI Colorbar", position=(35,5), vmin=evi_vis_params['min'], vmax=evi_vis_params['max'], vis_params=evi_vis_params)
    # else:
        # st.write('No Layer to Display')
        
    return Map.to_streamlit(height=600)

# Set full width layout
st.set_page_config(layout="wide", page_title="Pak-Afghan Shared Water Boundaries")

CSS = """
/* Navigation bar styles */
nav {
    display: flex;
    justify-content: space-around;
    align-items: center;
    height: 50px;
    background-color: gray;
    color: white;
    margin-top:-20px;
}
nav a {
    margin: 0 15px;
    text-decoration: none;
    color: white;
}
.sidebar .btn {
    display: block;
    width: 100%;
    margin-bottom: 0.5em;
}
</style>
"""
def navigation_bar():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""
        <nav>
            <a href="#home">Home</a>
            <a href="#about">About</a>
            <a href="#docs">Documentation</a>
        </nav>
        """, unsafe_allow_html=True)



# Main app layout
def main():
    # Navbar
    # navigation_bar()
    tab1, tab2, tab3 = st.tabs(["Home", "About", "Documentation"])

    st.write("#")
    # Initialize session state variables (dual charts here or replace below below set to ndvi_chart)
    if 'selected_basin' not in st.session_state:
        st.session_state['selected_basin'] = 'None'
    if 'selected_sub_basin' not in st.session_state:
        st.session_state['selected_sub_basin'] = 'None'
    if 'selected_index' not in st.session_state:  # Headache
        st.session_state['selected_index'] = 'None'
    if 'geojson_data' not in st.session_state:
        st.session_state['geojson_data'] = None
    if 'ndvi_image' not in st.session_state:
        st.session_state['ndvi_image'] = None
    if 'ndvi_chart' not in st.session_state:
        st.session_state['ndvi_chart'] = None
    if 'et_image' not in st.session_state:
        st.session_state['et_image'] = None
    if 'et_chart' not in st.session_state:
        st.session_state['et_chart'] = None
    if 'windspeed_image' not in st.session_state:
        st.session_state['windspeed_image'] = None
    if 'windspeed_chart' not in st.session_state:
        st.session_state['windspeed_chart'] = None
    
    if 'snow_cover_image' not in st.session_state:
        st.session_state['snow_cover_image'] = None
    if 'snow_cover_chart' not in st.session_state:
        st.session_state['snow_cover_chart'] = None

    if 'transpiration_image' not in st.session_state:
        st.session_state['transpiration_image'] = None
    if 'transpiration_chart' not in st.session_state:
        st.session_state['transpiration_chart'] = None

    if 'soil_moisture_image' not in st.session_state:
        st.session_state['soil_moisture_image'] = None
    if 'soil_moisture_chart' not in st.session_state:
        st.session_state['soil_moisture_chart'] = None
    
    if 'soil_temperature_image' not in st.session_state:
        st.session_state['soil_temperature_image'] = None
    if 'soil_temperature_chart' not in st.session_state:
        st.session_state['soil_temperature_chart'] = None

    if 'Precp_image' not in st.session_state:
        st.session_state['Precp_image'] = None
    if 'Precp_chart' not in st.session_state:
        st.session_state['Precp_chart'] = None

    if 'LAI_image' not in st.session_state:
        st.session_state['LAI_image'] = None
    if 'LAI_chart' not in st.session_state:
        st.session_state['LAI_chart'] = None
    if 'pet_image' not in st.session_state:
        st.session_state['pet_image'] = None
    if 'pet_chart' not in st.session_state:
        st.session_state['pet_chart'] = None
    if 'lhf_image' not in st.session_state:
        st.session_state['lhf_image'] = None
    if 'lhf_chart' not in st.session_state:
        st.session_state['lhf_chart'] = None
    if 'lst_image' not in st.session_state:
        st.session_state['lst_image'] = None
    if 'lst_chart' not in st.session_state:
        st.session_state['lst_chart'] = None
    if 'temp_image' not in st.session_state:
        st.session_state['temp_image'] = None
    if 'temp_chart' not in st.session_state:
        st.session_state['temp_chart'] = None
    if 'evi_image' not in st.session_state:
        st.session_state['evi_image'] = None
    if 'evi_chart' not in st.session_state:
        st.session_state['evi_chart'] = None
    if 'kbdi_image' not in st.session_state:
        st.session_state['kbdi_image'] = None
    if 'kbdi_chart' not in st.session_state:
        st.session_state['kbdi_chart'] = None
    if 'srtm_image' not in st.session_state:
        st.session_state['srtm_image'] = None
    if 'population_image' not in st.session_state:
        st.session_state['population_image'] = None
    if 'landcover_image' not in st.session_state:
        st.session_state['landcover_image'] = None
    if 'filter_options_visible' not in st.session_state:
        st.session_state['filter_options_visible'] = False
    if 'upload_section_visible' not in st.session_state:
        st.session_state['upload_section_visible'] = False
    if 'executed' not in st.session_state:
        st.session_state['executed'] = False
    if 'uploaded_roi' not in st.session_state:
        st.session_state['uploaded_roi'] = False

    # Sidebar content
    with st.sidebar:
        st.image('logo.png', width=300)
        cols = st.columns(3)
        with cols[0]:
            if st.button("Filter Basin"):
                st.session_state['filter_options_visible'] = True
                st.session_state['upload_section_visible'] = False
                st.session_state['uploaded_roi'] = False
        with cols[1]:
            # upload_roi = st.button("Upload ROI")
            if st.button("Upload ROI"):
                st.session_state['upload_section_visible'] = True
                # Hide the filter options when the upload section is shown
                st.session_state['filter_options_visible'] = False
                st.session_state['chart'] = None
                nullify_map()
                
        with cols[2]:
            draw_aoi = st.button("Draw AOI",disabled=True)
         # Display the file uploader if the upload section is visible
        if st.session_state.get('upload_section_visible', False):
            uploaded_file = st.file_uploader("Upload GeoJSON file", type=['geojson'], accept_multiple_files=False, key="geojson_upload")
            if uploaded_file is not None and uploaded_file.size <= 1_000_000:  # Check for file size (1 MB max)
                # Read the GeoJSON file
                geojson_data = json.load(uploaded_file)
                st.session_state['geojson_data'] = geojson_data
                st.session_state['uploaded_roi'] = True
            elif uploaded_file is not None:
                st.error("File too large. Please upload a file less than 1 MB.")
        if st.session_state.get('uploaded_roi', False):
            from_date_uploaded = st.date_input("From Date", key='from_date_uploaded')
            to_date_uploaded = st.date_input("To Date", key='to_date_uploaded')
            selected_index_uploaded = st.selectbox("Select Index", ['None', 'NDVI'], key='selected_index_uploaded')

            if st.button('Execute Uploaded ROI'):
                if selected_index_uploaded == 'NDVI':
                    # Convert the dates to strings in the format 'YYYY-MM-DD'
                    from_date_str = from_date_uploaded.strftime('%Y-%m-%d')
                    to_date_str = to_date_uploaded.strftime('%Y-%m-%d')

                    # Process the uploaded ROI
                    st.session_state['ndvi_image'] = get_ndvi_image_for_roi(st.session_state['geojson_data'], from_date_str, to_date_str)
                
                    # Indicate that the NDVI image should be displayed on the map
                    st.session_state['executed'] = True

        # Display filter options if the filter section is visible
        if st.session_state.get('filter_options_visible', False):
        # Sidebar filter options
            if st.session_state['filter_options_visible']:
                # with st.sidebar:
                selected_basin = st.selectbox("Select Basin", ['None'] + get_basins(), key='selected_basin')
                selected_sub_basin = st.selectbox("Select Sub-Basin", get_sub_basins(selected_basin), key='selected_sub_basin')
                from_date = st.date_input("From Date", datetime.today(), key='from_date')
                to_date = st.date_input("To Date", datetime.today(), key='to_date')
                selected_index = st.selectbox("Select Index", ['None', 'Air Temperature','Enhanced Vegetation Index (EVI)','Evapotranspiration ET',"Keetch-Byram Drought Index","Land Cover (2020)","Land Surface Temperature","Latent Heat Flux (LE)","Leaf Area Index","NDVI","Population","Potential Evapotranspiration (PET)","Precipitation","Snow Cover","Soil Moisture","Soil Temperature","Soil Types","SRTM","Transpiration (TP)","Wind Speed"], key='selected_index')

                execute_col, download_col = st.columns([1,1], gap="small")
            with execute_col:
                if st.button('Execute'):
                    # st.write(f"Selected Index before any operation: {st.session_state['selected_index']}")

                    # Reset images in session state
                    st.session_state['ndvi_image'] = None
                    st.session_state['srtm_image'] = None
                    st.session_state['temp_image'] = None
                    st.session_state['lst_image'] = None
                    st.session_state['evi_image'] = None
                    st.session_state['kbdi_image'] = None
                    st.session_state['et_image'] = None
                    st.session_state['landcover_image'] = None
                    st.session_state['lhf_image'] = None
                    st.session_state['pet_image'] = None
                    st.session_state['LAI_image'] = None
                    st.session_state['Precp_image'] = None
                    st.session_state['transpiration_image'] = None
                    st.session_state['snow_cover_image'] = None
                    st.session_state['population_image'] = None
                    st.session_state['soil_moisture_image'] = None
                    st.session_state['soil_temperature_image'] = None
                    st.session_state['windspeed_image'] = None
                    # Get the appropriate image based on selected index
                    if selected_index == 'Air Temperature':
                        st.session_state['temp_image'] = get_era_temp_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_temp_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        
                        csv = st.session_state['temp_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='airtemp_timeseries.csv',
                                mime='text/csv',
                                )
                        
                    if selected_index == 'Land Surface Temperature':
                        st.session_state['lst_image'] = get_lst_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_lst_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                       
                        csv = st.session_state['lst_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='lst_timeseries.csv',
                                mime='text/csv',
                                )
                    
                    if selected_index == 'Latent Heat Flux (LE)':
                        st.session_state['lhf_image'] = get_lhf_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_lhf_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                       
                        csv = st.session_state['lhf_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='lhf_timeseries.csv',
                                mime='text/csv',
                                )
                    if selected_index == 'Potential Evapotranspiration (PET)':
                        st.session_state['pet_image'] = get_pet_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_pet_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                       
                        csv = st.session_state['pet_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='pet_timeseries.csv',
                                mime='text/csv',
                                )
                    if selected_index == 'Leaf Area Index':
                        st.session_state['LAI_image'] = get_LAI_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_LAI_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        
                        csv = st.session_state['LAI_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='LAI_timeseries.csv',
                                mime='text/csv',
                                )

                    if selected_index == 'Precipitation':
                        st.session_state['Precp_image'] = get_Precp_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_Precp_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        
                        csv = st.session_state['Precp_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='Precp_timeseries.csv',
                                mime='text/csv',
                                ) 
                    
                    if selected_index == 'Soil Moisture':
                        st.session_state['soil_moisture_image'] = get_soil_moisture_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_soil_moisture_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        
                        csv = st.session_state['soil_moisture_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='soil_moisture_timeseries.csv',
                                mime='text/csv',
                                )
                    if selected_index == 'Soil Temperature':
                        st.session_state['soil_temperature_image'] = get_soil_temperature_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_soil_temperature_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        
                        csv = st.session_state['soil_temperature_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='soil_temperature_timeseries.csv',
                                mime='text/csv',
                                )
                    
                    if selected_index == 'Transpiration (TP)':
                        st.session_state['transpiration_image'] = get_transpiration_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_transpiration_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        
                        csv = st.session_state['transpiration_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='transpiration_timeseries.csv',
                                mime='text/csv',
                                )
                    
                    if selected_index == 'Wind Speed':
                        st.session_state['windspeed_image'] = get_windspeed_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_windspeed_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        
                        csv = st.session_state['windspeed_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='windspeed_timeseries.csv',
                                mime='text/csv',
                                )

                    if selected_index == 'Snow Cover':
                        st.session_state['snow_cover_image'] = get_snow_cover_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_snow_cover_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        
                        csv = st.session_state['snow_cover_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='snow_cover_timeseries.csv',
                                mime='text/csv',
                                )
    
                    if selected_index == 'Keetch-Byram Drought Index': 
                        st.session_state['kbdi_image'] = get_kbdi_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_kbdi_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        csv = st.session_state['kbdi_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='kbdi_timeseries.csv',
                                mime='text/csv',
                                )
                    
                    if selected_index == 'Evapotranspiration ET':
                        st.session_state['et_image'] = get_et_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_et_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        csv = st.session_state['et_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='et_timeseries.csv',
                                mime='text/csv',
                                )
                    
                    if selected_index == 'Enhanced Vegetation Index (EVI)':
                        st.session_state['evi_image'] = get_evi_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_evi_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        csv = st.session_state['evi_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='evi_timeseries.csv',
                                mime='text/csv',
                                )
                    if selected_index == 'SRTM':
                        st.session_state['srtm_image'] = get_srtm_image(selected_sub_basin)
                    if selected_index == 'Population':
                        st.session_state['population_image'] = get_population_image(selected_sub_basin)
                    elif selected_index == 'Land Cover (2020)':
                        st.session_state['landcover_image'] = get_landcover_image(selected_sub_basin)
                    if selected_index == 'NDVI':
                        st.session_state['ndvi_image'] = get_ndvi_image(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        st.session_state['chart'] = create_ndvi_timeseries(selected_sub_basin, st.session_state['from_date'].strftime('%Y-%m-%d'), st.session_state['to_date'].strftime('%Y-%m-%d'))
                        # st.pyplot(st.session_state['ndvi_chart']) #show chart in sidebar
                        # Convert chart data to CSV
                        csv = st.session_state['ndvi_chart_data'].to_csv(index=False)
                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name='ndvi_timeseries.csv',
                                mime='text/csv',
                                )
                        
                    st.session_state['executed'] = True
                    # st.write("Execution complete, session state updated.")
                    
            with download_col:
                if st.session_state['executed']:
                    if st.button(f'Save GeoTIFF'): #if st.button(f'Save {selected_index}'):
                        # Code to download the image
                        try:
                            if selected_index == 'SRTM':
                                download_url = st.session_state['srtm_image'].getDownloadURL({'scale': 90})
                            elif selected_index == 'NDVI':
                                download_url = st.session_state['ndvi_image'].getDownloadURL({'scale': 250})
                                


                            st.markdown(f"[Download {selected_index}]({download_url})", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")



    # Display the map in the main area
    display_map(
        selected_index=st.session_state['selected_index'],
        selected_basin=st.session_state['selected_basin'],
        selected_sub_basin=st.session_state['selected_sub_basin'],
        ndvi_image=st.session_state['ndvi_image'],
        temp_image=st.session_state['temp_image'],
        evi_image=st.session_state['evi_image'],
        srtm_image=st.session_state['srtm_image'],
        lst_image=st.session_state['lst_image'],
        lhf_image=st.session_state['lhf_image'],
        landcover_image=st.session_state['landcover_image'],
        et_image=st.session_state['et_image'],
        pet_image=st.session_state['pet_image'],
        kbdi_image=st.session_state['kbdi_image'],
        LAI_image=st.session_state['LAI_image'],
        Precp_image=st.session_state['Precp_image'],
        transpiration_image=st.session_state['transpiration_image'],
        soil_moisture_image=st.session_state['soil_moisture_image'],
        soil_temperature_image=st.session_state['soil_temperature_image'],
        snow_cover_image=st.session_state['snow_cover_image'],
        population_image=st.session_state['population_image'],
        windspeed_image=st.session_state['windspeed_image'],
        geojson_data=st.session_state.get('geojson_data') 

        # st.session_state.get('geojson_data')
    )

    if 'chart' in st.session_state and st.session_state['chart'] is not None:
        st.pyplot(st.session_state['chart'])

    # if 'ndvi_chart' in st.session_state and st.session_state['ndvi_chart'] is not None:
    #     st.pyplot(st.session_state['ndvi_chart'])
    # if 'temp_chart' in st.session_state and st.session_state['temp_chart'] is not None:
    #     st.pyplot(st.session_state['temp_chart'])
    # if 'evi_chart' in st.session_state and st.session_state['evi_chart'] is not None:
    #     st.pyplot(st.session_state['evi_chart'])
    # if 'et_chart' in st.session_state and st.session_state['et_chart'] is not None:
    #     st.pyplot(st.session_state['et_chart'])
    # if 'kbdi_chart' in st.session_state and st.session_state['kbdi_chart'] is not None:
    #     st.pyplot(st.session_state['kbdi_chart'])
    # if 'lst_chart' in st.session_state and st.session_state['lst_chart'] is not None:
    #     st.pyplot(st.session_state['lst_chart'])
    # if 'et_chart' in st.session_state and st.session_state['et_chart'] is not None:
    #     st.pyplot(st.session_state['et_chart'])
    # if 'lhf_chart' in st.session_state and st.session_state['lhf_chart'] is not None:
    #     st.pyplot(st.session_state['lhf_chart'])
    # if 'pet_chart' in st.session_state and st.session_state['pet_chart'] is not None:
    #     st.pyplot(st.session_state['pet_chart'])
    # if 'LAI_chart' in st.session_state and st.session_state['LAI_chart'] is not None:
    #     st.pyplot(st.session_state['LAI_chart'])
    # if 'Precp_chart' in st.session_state and st.session_state['Precp_chart'] is not None:
    #     st.pyplot(st.session_state['Precp_chart'])
    # if 'snow_cover_chart' in st.session_state and st.session_state['snow_cover_chart'] is not None:
    #     st.pyplot(st.session_state['snow_cover_chart'])
    # if 'soil_moisture_chart' in st.session_state and st.session_state['soil_moisture_chart'] is not None:
    #     st.pyplot(st.session_state['soil_moisture_chart'])
    # if 'soil_temperature_chart' in st.session_state and st.session_state['soil_temperature_chart'] is not None:
    #     st.pyplot(st.session_state['soil_temperature_chart'])
    # if 'transpiration_chart' in st.session_state and st.session_state['transpiration_chart'] is not None:
    #     st.pyplot(st.session_state['transpiration_chart'])
    # if 'windspeed_chart' in st.session_state and st.session_state['windspeed_chart'] is not None:
    #     st.pyplot(st.session_state['windspeed_chart'])
                
        

# Run the app
if __name__ == "__main__":
    main()

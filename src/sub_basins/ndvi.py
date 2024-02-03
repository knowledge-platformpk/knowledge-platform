import streamlit as st
import ee
import pandas as pd
import matplotlib.pyplot as plt

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

    minMax = mean_ndvi_image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=sub_basin_feature,
        scale=30,
        bestEffort=True
    )

    # Access min and max values from the minMax dictionary
    try:
        min_image = minMax.get('NDVI_min')
        max_image = minMax.get('NDVI_max')
    
        st.session_state['min'] = min_image.getInfo()
        st.session_state['max'] = max_image.getInfo()
    except:
        st.session_state['min'] = 0
        st.session_state['max'] = 1

    sub_basin_geometry = sub_basin_feature.geometry()
    mean_ndvi_image_region = ndvi_collection.mean().clipToBoundsAndScale(geometry=sub_basin_geometry, maxDimension=1000)

    return mean_ndvi_image, mean_ndvi_image_region

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

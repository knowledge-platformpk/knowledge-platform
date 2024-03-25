import streamlit as st
import ee
import pandas as pd
import matplotlib.pyplot as plt

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

    minMax = mean_pet_image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=sub_basin_feature,
        scale=30,
        bestEffort=True
    )
    try:
    # Access min and max values from the minMax dictionary
        min_image = minMax.get('PET_min')
        max_image = minMax.get('PET_max')

        st.session_state['min'] = min_image.getInfo()
        st.session_state['max'] = max_image.getInfo()
    except:
        st.session_state['min'] = 0
        st.session_state['max'] = 75

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
    plt.ylabel('Potential Evapotranspiration (PET) kg/m^2/8day')
    plt.grid(True)
    plt.tight_layout()

    return fig

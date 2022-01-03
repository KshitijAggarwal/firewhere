import json
import urllib

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


@st.cache(show_spinner=False)
def get_dict_from_url(url):
    """

    Args:
        url: URL to read the JSON from.

    Returns:

    """
    response = urllib.request.urlopen(url)
    return json.loads(response.read())


@st.cache(show_spinner=False, persist=True)
def get_counties():
    """
    Reads the file with county locations and returns the dictionary.

    """
    PATH = "https://firewhere-data.s3.us-east-2.amazonaws.com/data/"
    return get_dict_from_url(f"{PATH}countyinfo.json")


def get_county_loc(counties):
    """
    Returns the location of county.

    Args:
        counties: dictionary containing the county info.

    Returns:

    """
    c1, c2 = st.sidebar.columns(2)
    with c1:
        state = st.sidebar.selectbox("State", set(counties.keys()))
    with c2:
        county = st.sidebar.selectbox("County", counties[state])

    loc = counties[state][county]
    return loc["Latitude"], loc["Longitude"]


def check_doy(doy):
    """
    Makes sure the day of year is between 1 and 365.

    """
    if doy < 1 or doy > 365:
        st.error("Day of year has to be between 1 and 365.")
        return None
    else:
        return 1


@st.cache(show_spinner=False, persist=True)
def read_weather_data():
    """
    Reads 30-year averaged weather data

    Returns:
        DataFrame with weather data.

    """
    # st.write('read weather data function is running. ')
    PATH = "https://firewhere-data.s3.us-east-2.amazonaws.com/data/"
    return pd.read_parquet(f"{PATH}combined.parquet")


@st.cache(show_spinner=False, persist=True)
def read_stations():
    """
    Reads station information and returns a DataFrame

    """
    # st.write('read stations function is running.')
    PATH = "https://firewhere-data.s3.us-east-2.amazonaws.com/data/"
    return pd.read_parquet(f"{PATH}common_stations.parquet")


@st.cache(show_spinner=False)
def get_weather_params(lat, long, doy, common_stations, temp_data):
    """
    Return weather values for a given location and day of year.
    It searches for the nearest weather station and returns the values from it.

    Args:
        lat: Latitude
        long: Longitude
        doy: Day of the year
        common_stations: Weather station names
        temp_dat: Average temperature, variation, precipitation and snowfall

    Returns:
        Temperature, Diurnal, Precipitation and Snow values for that location and day.

    """
    dist = np.array(
        np.sqrt(
            (common_stations["lat"] - lat) ** 2 + (common_stations["long"] - long) ** 2
        )
    )
    ind = np.argmin(dist)
    if dist[ind] > 1:
        st.error(
            "No weather stations within 1 degrees of the queried location. Try some other location. "
            "Approximate lat/long ranges for US are: "
            "Latitude: 30 to 50, Longitude: -70 to -120."
        )

        return None
    cst = common_stations.iloc[ind]
    station_id = cst["id"]
    doy = int(doy)
    temp_val, dutr_val, prcp_val, snow_val = temp_data[station_id][doy]
    return temp_val, dutr_val, prcp_val, snow_val


@st.cache(allow_output_mutation=True)
def load_model():
    """
    Load the model.

    Returns:

    """
    MODEL_PATH = "https://firewhere-data.s3.us-east-2.amazonaws.com/model.tar.gz"
    local_path = tf.keras.utils.get_file("model", MODEL_PATH, extract=True)
    p = "/".join(local_path.split("/")[:-1])
    return tf.keras.models.load_model(p + "/trained_model")

import json
import urllib

import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd

#PATH = "https://firewhere-data.s3.us-east-2.amazonaws.com/data/"
MODEL_PATH = "https://firewhere-data.s3.us-east-2.amazonaws.com/model.tar.gz"


@st.cache(show_spinner=False)
def get_dict_from_url(url):
    """

    Args:
        url: URL to read the JSON from.

    Returns:

    """
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


@st.cache(show_spinner=False, persist=True)
def get_counties(PATH):
    """
    Reads the file with county locations and returns the dictionary.

    """
    st.write('read counties function is running. ')
    counties = get_dict_from_url(f"{PATH}countyinfo.json")
    return counties


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
def read_weather_data(PATH):
    """
    Read weather data from individual files.

    Returns:
        Dictionaries with weather data.

    """
    st.write('read weather data function is running. ')
    tavg = get_dict_from_url(f"{PATH}tavg.json")
    diur = get_dict_from_url(f"{PATH}diur.json")
    snow = get_dict_from_url(f"{PATH}snow.json")
    prcp = get_dict_from_url(f"{PATH}prcp.json")
    return tavg, diur, prcp, snow

@st.cache(show_spinner=False, persist=True)
def read_stations(PATH):
    """

    Returns:

    """
    st.write('read stations function is running.')
    return pd.read_csv(f"{PATH}common_stations.csv")


@st.cache(show_spinner=False)
def get_weather_params(lat, long, doy, common_stations, tavg, diur, prcp, snow):
    """
    Return weather values for a given location and day of year. It searches for the nearest weather station and
    returns the values from it.

    Args:
        lat: Latitude
        long: Longitude
        doy: Day of the year
        common_stations: Weather station names
        tavg: Average temperature
        diur: Average temperature difference between the minimum at night (low) and the maximum during the day (high).
        prcp: Average precipitation.
        snow: Average snowfall.

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
            "Latitude: 30 to 50, Logitude: -70 to -120."
        )

        return None
    cst = common_stations.iloc[ind]
    station_id = cst["id"]
    doy = str(float(doy))
    temp_val = tavg[station_id][doy]
    dutr_val = diur[station_id][doy]
    prcp_val = prcp[station_id][doy]
    snow_val = snow[station_id][doy]
    return temp_val, dutr_val, prcp_val, snow_val


# @st.cache(allow_output_mutation=True)
def load_model():
    """
    Load the model.

    Returns:

    """
    local_path = tf.keras.utils.get_file("model", MODEL_PATH, extract=True)
    p = "/".join(local_path.split("/")[:-1])
    return tf.keras.models.load_model(p + "/trained_model")

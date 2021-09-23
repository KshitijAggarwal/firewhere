import json

import numpy as np
import streamlit as st
import tensorflow as tf

PATH = "/home/kshitij/firewhere/useful_data/data/"


def get_county_loc():
    """

    Returns:

    """
    with open(f"{PATH}countyinfo.json", "r") as fp:
        counties = json.load(fp)
    state = st.selectbox("State", set(counties.keys()))
    county = st.selectbox("County", counties[state])

    loc = counties[state][county]
    #     if st.button('Selected'):
    #         st.write(f'Selected Values: {state}, {county}, {loc}')
    return loc["Latitude"], loc["Longitude"]


def check_doy(doy):
    """

    Args:
        doy:

    Returns:

    """
    if doy < 1 or doy > 365:
        st.error("Day of year has to be between 1 and 365.")
        return None
    else:
        return 1


def read_weather_data():
    """

    Returns:

    """
    with open(f"{PATH}tavg.json", "r") as fp:
        tavg = json.load(fp)
    with open(f"{PATH}diur.json", "r") as fp:
        diur = json.load(fp)
    with open(f"{PATH}prcp.json", "r") as fp:
        prcp = json.load(fp)
    with open(f"{PATH}snow.json", "r") as fp:
        snow = json.load(fp)
    return tavg, diur, prcp, snow


def get_weather_params(lat, long, doy, common_stations, tavg, diur, prcp, snow):
    """

    Args:
        lat:
        long:
        doy:
        common_stations:
        tavg:
        diur:
        prcp:
        snow:

    Returns:

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


def load_model():
    """

    Returns:

    """
    return tf.keras.models.load_model("/home/kshitij/firewhere/useful_data/fw_model")

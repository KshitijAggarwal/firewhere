import json

import numpy as np
import streamlit as st
import tensorflow as tf
import urllib

PATH = "https://firewhere-data.s3.us-east-2.amazonaws.com/data/"
MODEL_PATH = "https://firewhere-data.s3.us-east-2.amazonaws.com/model.tar.gz"


@st.cache()
def get_dict_from_url(url):
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


@st.cache()
def get_counties():
    """

    Returns:

    """

    counties = get_dict_from_url(f"{PATH}countyinfo.json")
    return counties


def get_county_loc(counties):
    """

    Args:
        counties:

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

    Args:
        doy:

    Returns:

    """
    if doy < 1 or doy > 365:
        st.error("Day of year has to be between 1 and 365.")
        return None
    else:
        return 1


# @st.cache
@st.cache()
def read_weather_data():
    """

    Returns:

    """
    tavg = get_dict_from_url(f"{PATH}tavg.json")
    diur = get_dict_from_url(f"{PATH}diur.json")
    snow = get_dict_from_url(f"{PATH}snow.json")
    prcp = get_dict_from_url(f"{PATH}prcp.json")
    return tavg, diur, prcp, snow


@st.cache()
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


# @st.cache(allow_output_mutation=True)
def load_model():
    """

    Returns:

    """
    local_path = tf.keras.utils.get_file("model", MODEL_PATH, extract=True)
    p = "/".join(local_path.split("/")[:-1])
    return tf.keras.models.load_model(p + "/trained_model")

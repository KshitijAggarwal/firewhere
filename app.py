import json
import numpy as np
import pandas as pd
import streamlit as st

from stutils import *

# import tensorflow as tf

act_mapping = {
    "Lightening": "0",
    "Equipment Use": "1",
    "Smoking": "2",
    "Campfire": "3",
    "Debris Burning": "4",
    "Railroad": "5",
    "Arson": "6",
    "Children": "7",
    "Miscellaneous": "8",
    "Fireworks": "9",
    "Powerline": "10",
    "Structure": "11",
    "Missing/Undefined": "12",
}

activity = st.selectbox("Human activity", act_mapping)

# st.write('Zip to lat long using https://simplemaps.com/data/us-counties')

act_index = float(act_mapping[activity])

inp = st.radio("Input Options", ("Lat./Lon.", "County", "Zipcode"))

if inp == "Lat./Lon.":
    lat = st.number_input(label="Latitude", step=1.0, format="%.2f")
    long = st.number_input(label="Longitude", step=1.0, format="%.2f")
elif inp == "County":
    lat, long = get_county_loc()
else:
    st.error("Not implemented")

doy = st.number_input(label="Day of Year", step=1)
weather_bool = st.checkbox("Show weather data")

if st.button("Predict Size"):
    check_doy(doy)
    check_pos(lat, long)
    #     model = load_model()
    try:
        temp_val, dutr_val, prcp_val, snow_val = get_weather_params(
            lat, long, doy, common_stations, tavg, diur, prcp, snow
        )
    except NameError:
        tavg, diur, prcp, snow = read_weather_data()
        common_stations = pd.read_csv("common_stations.csv")
        temp_val, dutr_val, prcp_val, snow_val = get_weather_params(
            lat, long, doy, common_stations, tavg, diur, prcp, snow
        )

    if weather_bool:
        st.write(f"Average values of weather parameters:")
        st.write(f"Temperature: {temp_val}")
        st.write(f"Diurnal air temperature variation: {dutr_val}")
        st.write(f"Precipitation: {prcp_val}")
        st.write(f"Snow: {snow_val}")

    inp = np.array([lat, long, act_index, temp_val, dutr_val, prcp_val, snow_val])
    st.write(inp)
#     inp = tf.convert_to_tensor(np.expand_dims(inp, 0))

#     res = model.predict(inp)
#     st.write(f"Predicted firesize is {10**res[0][0]}acres")
#     st.write(f'Selected Values: {activity}, {act_index}, {lat}, {long}, {doy}, {weather_bool}')

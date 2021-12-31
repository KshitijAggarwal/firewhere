import folium
from streamlit_folium import folium_static

from stutils import *

PATH = "https://firewhere-data.s3.us-east-2.amazonaws.com/data/"

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


def main():
    # Human activity and day of year selection
    cact, cdoy = st.sidebar.columns(2)
    with cact:
        activity = st.sidebar.selectbox("Human activity", act_mapping)
    with cdoy:
        doy = st.sidebar.number_input(label="Day of Year", step=1)

    act_index = float(act_mapping[activity])

    inp = st.sidebar.radio("Input Options", ("Lat./Lon.", "County"))

    # Location selection to get lat. and lon.
    if inp == "Lat./Lon.":
        clat, clong = st.sidebar.columns(2)
        with clat:
            lat = st.sidebar.number_input(label="Latitude", step=1.0, format="%.2f")
        with clong:
            long = st.sidebar.number_input(label="Longitude", step=1.0, format="%.2f")
    else:
        counties = get_counties(PATH)
        lat, long = get_county_loc(counties)

    # To display weather data and location.
    c1, c2 = st.columns(2)
    with c1:
        weather_bool = st.checkbox("Show weather data")
    with c2:
        show_map = st.checkbox("Show location on map")

    # Predict button
    if st.button("Predict Size"):
        if not check_doy(doy):
            return None

        with st.spinner("Setting up lookup tables"):
            tavg, diur, prcp, snow = read_weather_data(PATH)
            common_stations = read_stations(PATH)

        with st.spinner("Reading weather parameters"):
            vals = get_weather_params(
                lat, long, doy, common_stations, tavg, diur, prcp, snow
            )

        temp_val, dutr_val, prcp_val, snow_val = vals

        if weather_bool:
            html_str = f"""
                        ### Average values of weather parameters:
                            * Temperature: {temp_val}
                            * Diurnal air temperature variation: {dutr_val}
                            * Precipitation: {prcp_val}
                            * Snow: {snow_val}
                        """
            st.markdown(html_str, unsafe_allow_html=True)

        if show_map:
            m = folium.Map(location=[lat, long], zoom_start=16, tiles="OpenStreetMap")

            folium.Marker([lat, long], popup="Location").add_to(m)
            folium_static(m)

        with st.spinner("Running ML model"):
            inp = np.array(
                [lat, long, doy, act_index, temp_val, dutr_val, prcp_val, snow_val]
            )
            inp = tf.convert_to_tensor(np.expand_dims(inp, 0))

            model = load_model()
            res = model.predict(inp)

            st.markdown(f"### Predicted firesize is {10 ** res[0][0]:.3f} acres")


if __name__ == "__main__":
    main()

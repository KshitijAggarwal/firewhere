import datetime

import folium
from streamlit_folium import folium_static

from stutils import *

act_mapping = {
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
    "Lightening": "0",
}


def main():
    # Human activity and day of year selection
    cact, cdoy = st.sidebar.columns(2)
    with cact:
        activity = st.sidebar.selectbox("Cause of the fire", act_mapping)
    with cdoy:
        day = st.sidebar.date_input(
            "Date", datetime.datetime.now(), min_value=datetime.datetime.now()
        )

    act_index = float(act_mapping[activity])

    inp = st.sidebar.radio("Location Options", ("County", "Lat./Lon."))

    doy = day.timetuple().tm_yday
    # Location selection to get lat. and lon.
    if inp == "Lat./Lon.":
        clat, clong = st.sidebar.columns(2)
        with clat:
            lat = st.sidebar.number_input(label="Latitude", step=1.0, format="%.2f")
        with clong:
            long = st.sidebar.number_input(label="Longitude", step=1.0, format="%.2f")
    else:
        counties = get_counties()
        lat, long = get_county_loc(counties)

    st.markdown("# FireWhere")

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
            temp_data = read_weather_data()
            common_stations = read_stations()

        with st.spinner("Reading weather parameters"):
            temp_val, dutr_val, prcp_val, snow_val = get_weather_params(
                lat, long, doy, common_stations, temp_data
            )

        with st.spinner("Running ML model"):
            inp = np.array(
                [lat, long, doy, act_index, temp_val, dutr_val, prcp_val, snow_val]
            )
            inp = tf.convert_to_tensor(np.expand_dims(inp, 0))

            model = load_model()
            res = model.predict(inp)

            arg = np.argmax(res)

            if arg == 0:
                s = "less than 0.22 acres"
            elif arg == 1:
                s = "between 0.22 and 2 acres"
            else:
                s = "greater than 2 acres"

            st.markdown(f"### Predicted fire size is {s}")

        if weather_bool:
            html_str = f"""
                        #### Average values of weather parameters:
                            * Temperature: {temp_val}F
                            * Daily temperature variation: {dutr_val}F
                            * Precipitation: {prcp_val}mm
                            * Snow: {snow_val}mm
                        """
            st.markdown(html_str, unsafe_allow_html=True)

        if show_map:
            m = folium.Map(location=[lat, long], zoom_start=16, tiles="OpenStreetMap")

            folium.Marker([lat, long], popup="Location").add_to(m)
            folium_static(m)


if __name__ == "__main__":
    main()

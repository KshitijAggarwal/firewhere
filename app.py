import folium
import pandas as pd
from streamlit_folium import folium_static

from stutils import *

PATH = "/home/kshitij/firewhere/useful_data/data/"

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
    activity = st.selectbox("Human activity", act_mapping)
    act_index = float(act_mapping[activity])

    inp = st.radio("Input Options", ("Lat./Lon.", "County"))

    if inp == "Lat./Lon.":
        lat = st.number_input(label="Latitude", step=1.0, format="%.2f")
        long = st.number_input(label="Longitude", step=1.0, format="%.2f")
    else:
        lat, long = get_county_loc()

    doy = st.number_input(label="Day of Year", step=1)
    c1, c2 = st.columns(2)
    with c1:
        weather_bool = st.checkbox("Show weather data")
    with c2:
        show_map = st.checkbox("Show location on map")

    if st.button("Predict Size"):
        if not check_doy(doy):
            return None

        with st.spinner("Reading weather parameters"):
            try:
                vals = get_weather_params(
                    lat, long, doy, common_stations, tavg, diur, prcp, snow
                )
            except NameError:
                tavg, diur, prcp, snow = read_weather_data()
                common_stations = pd.read_csv(f"{PATH}common_stations.csv")
                vals = get_weather_params(
                    lat, long, doy, common_stations, tavg, diur, prcp, snow
                )

            if vals:
                temp_val, dutr_val, prcp_val, snow_val = vals
            else:
                return None

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
                # loc = pd.DataFrame({'lat': [lat], 'lon': [long]})
                # st.map(loc, zoom=8)
                m = folium.Map(
                    location=[lat, long], zoom_start=16, tiles="OpenStreetMap"
                )  # tiles="Stamen Terrain"

                folium.Marker([lat, long], popup="Location").add_to(m)
                folium_static(m)

        with st.spinner("Running ML model"):
            inp = np.array(
                [lat, long, act_index, temp_val, dutr_val, prcp_val, snow_val]
            )
            inp = tf.convert_to_tensor(np.expand_dims(inp, 0))

            try:
                res = model.predict(inp)
            except NameError:
                model = load_model()
                res = model.predict(inp)

            st.markdown(f"### Predicted firesize is {10 ** res[0][0]:.3f} acres")

if __name__ == "__main__":
    main()

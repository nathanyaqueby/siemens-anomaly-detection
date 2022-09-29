import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events


st.set_page_config(layout="wide")
st.title("Anomaly Detection Team - Challenge 4")
st.markdown("Upload data to start!")


st.sidebar.title("1. Data")
uploaded_file = st.sidebar.file_uploader("Choose a file (CTR, HMI, or SEA as Excel file)")

if uploaded_file is not None:

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_excel(uploaded_file, sheet_name="Sheet1")

    df_type = uploaded_file.name.split("_")[0]
    df = dataframe
    val = "Value"
    date = "Date"

    if df_type in ["CTR", "HMI"]:
        lcb = "LCB "
    else:
        lcb = "LCB"

    # st.sidebar.checkbox("Show Analysis by Location", True, key=1)
    st.sidebar.title("2. Location")
    select = st.sidebar.selectbox('Select a Location', df[lcb])

    #get the state selected in the selectbox
    state_data = df[df[lcb] == select]

    countries=df[lcb].unique()
    dic = {}
    for  country in countries:
        dic[country]=df[df[lcb]==country]

    def get_total_dataframe(dataset):
        total_dataframe = pd.DataFrame({
        'Date':dataset[date],
        'Value':dataset[val]})
        return total_dataframe

    state_total = get_total_dataframe(state_data)
    # state_total = ctr_data

    if st.sidebar.checkbox("Show Analysis by Location", True, key=2):
        st.markdown("## **Location analysis**")
        st.markdown(f"### Overall {df_type} data in {select} from October 2020 to last week")
        if not st.checkbox('Hide Graph', False, key=1):
            # state_total_graph = px.line(
            # state_total, 
            # x='Date',
            # y='Value',
            # labels={'Value':'Value in %s' % (select)},
            # width=1200, height=400)
            # st.plotly_chart(state_total_graph, use_container_width=True)
            # Writes a component similar to st.write()
            fig = px.line(
            state_total, 
            x='Date',
            y='Value',
            labels={'Value':'Value in %s' % (select)},
            width=1200, height=400)
            st.plotly_chart(fig, use_container_width=True)
            selected_points = plotly_events(fig)

            # # Can write inside of things using with!
            # with st.expander('Plot'):
            #     fig = px.line(
            #     state_total, 
            #     x='Date',
            #     y='Value',
            #     labels={'Value':'Value in %s' % (select)},
            #     width=1200, height=400)
            #     st.plotly_chart(fig, use_container_width=True)
            #     selected_points = plotly_events(fig)

            st.write(selected_points["x"],selected_points["y"])
            st.markdown("Selected data point, {}:{}".format(selected_points["x"],selected_points["y"]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(layout="wide")
st.title("Anomaly Detection Team - Challenge 4")
st.markdown("Upload data to start!")


st.sidebar.title("1. Data")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_excel(uploaded_file)
    st.write(dataframe)

    df_type = uploaded_file.name.split("_")[0]
    df = dataframe

    # st.sidebar.checkbox("Show Analysis by Location", True, key=1)
    st.sidebar.title("2. Location")
    select = st.sidebar.selectbox('Select a Location', df['LCB '])

    #get the state selected in the selectbox
    state_data = df[df['LCB '] == select]

    countries=df['LCB '].unique()
    dic={}
    for  country in countries:
        dic[country]=df[df['LCB ']==country]

    def get_total_dataframe(dataset):
        total_dataframe = pd.DataFrame({
        'Date':dataset['Date'],
        'Value':dataset['Value']})
        return total_dataframe

    state_total = get_total_dataframe(state_data)
    # state_total = ctr_data

    if st.sidebar.checkbox("Show Analysis by Location", True, key=2):
        st.markdown("## **Location analysis**")
        st.markdown(f"### Overall {df_type} data in {select} from October 2020 to last week")
        if not st.checkbox('Hide Graph', False, key=1):
            state_total_graph = px.line(
            state_total, 
            x='Date',
            y='Value',
            labels={'Value':'Value in %s' % (select)},
            width=1200, height=400)
            st.plotly_chart(state_total_graph, use_container_width=True)
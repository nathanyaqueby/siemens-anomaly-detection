import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF

st.set_page_config(layout="wide")
st.title("Anomaly Detection Team - Challenge 4")

if uploaded_file is None:
    st.markdown("Upload data to start!")

st.sidebar.title("1. Data")
uploaded_file = st.sidebar.file_uploader("Choose a file")

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
    select = st.sidebar.selectbox('Select a location', df[lcb])

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

    if st.sidebar.checkbox("Show analysis by location", True, key=2):
        st.markdown("## **Location analysis**")
        st.markdown(f"### Overall {df_type} data in {select} from October 2020 to last week")
        if not st.checkbox('Hide graph', False, key=1):
            state_total_graph = px.line(
            state_total, 
            x='Date',
            y='Value',
            labels={'Value':'Value in %s' % (select)},
            width=1200, height=400)
            st.plotly_chart(state_total_graph, use_container_width=True)
            pio.write_image(state_total_graph, "fig1.png", format="png", validate="False", engine="kaleido")


        # download as PDF
        pdf = FPDF('P', 'mm', 'A4')
        pdf.add_page()
        pdf.set_font(family='Arial', size=16)
        pdf.cell(40, 50, txt="Anomaly Detection Report")
        # pdf.cell(40, 50, txt=f"Overall {df_type} data in {select} from October 2020 to last week")
        pdf.image("fig1.png", w=195, h=65, y=40, x=10)

        st.download_button('Download report as PDF',
                        data=pdf.output(dest="S").encode("latin-1"),
                        file_name='anomaly_detection_report.pdf'
                        )

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide")
st.title("Anomaly Detection Team - Challenge 4")

st.sidebar.title("1. Data")
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel)")

if uploaded_file is None:
    st.markdown("Upload data to start!")

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
    for country in countries:
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
        date_min=df.Date.iloc[0].strftime("%B %Y")
        date_max=df.Date.iloc[-1].strftime("%B %Y")
        st.markdown(f"### Overall {df_type} data in {select} from {date_min} to {date_max}")
        if not st.checkbox('Hide graph', False, key=1):
            fig = px.line(
            state_total, 
            x='Date',
            y='Value',
            labels={'Value':'Value in %s' % (select)},
            width=1200, height=400)
            # create list of dicts with selected points, and plot
            selected_points = plotly_events(fig)
            # unsure why?
            pio.write_image(fig, "fig1.png", format="png", validate="False", engine="kaleido")
            # if a point was clicked, show info
            if selected_points:
                st.markdown("#### **Selected point**")
                st.markdown("Date: {}".format(selected_points[0]["x"]))
                st.markdown("Value: {}".format(selected_points[0]["y"]))
    else:
        fig = px.line(df, x='Date', y='Value', color=lcb,title=df_type)
        # fig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

    # download as PDF
    st. markdown("### **Save to pdf**")
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

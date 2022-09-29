import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF
from streamlit_plotly_events import plotly_events
import os
import statsmodels.tsa.stattools as sta
from prophet.serialize import model_to_json, model_from_json


def test_stationarity(ts_data, column='', signif=0.05, series=False):
    if series:
        adf_test = sta.adfuller(ts_data, autolag='AIC')
    else:
        adf_test = sta.adfuller(ts_data[column], autolag='AIC')
    p_value = adf_test[1]
    if p_value <= signif:
        test_result = "Stationary"
    else:
        test_result = "Non-Stationary"
    return test_result

def predict_model(m_path, df, country):

    dic={}
    dic[country]=df[df['LCB ']==country]
    dic[country].Label=dic[country].Label.replace(np.nan,'normal')
    dic[country].drop('Segment',inplace=True,axis=1)
    dic[country].drop('LCB ',inplace=True,axis=1)
    dic[country].drop('Label',inplace=True,axis=1)
    dic[country]=dic[country].rename(columns = {"Date":"ds","Value":"y"})

    with open(m_path, "r") as fin:
        m = model_from_json(fin.read())

    dataframe = dic[country]
    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].reset_index(drop = True)

    result = pd.concat([dataframe.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]], axis=1)
    result['error'] = result['y'] - result['yhat']
    result['uncertainty'] = result['yhat_upper'] - result['yhat_lower']
    result['anomaly'] = result.apply(lambda x: 'Yes' if(np.abs(x['error']) > 1.5*x['uncertainty']) else 'No', axis = 1)
    fig = px.scatter(result.reset_index(), x='ds', y='y', color='anomaly', title=country)
    pio.write_image(fig, "fig2.png", format="png", validate="False", engine="kaleido")

    # slider
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                dict(count=1, label='1y', step="year", stepmode="backward"),
                dict(count=2, label='3y', step="year", stepmode="backward"),
                dict(count=2, label='5y', step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    return dic, forecast, result

def analyze_data(df, dic, select, pred, result):
    evaluation={}
    dic1={}
    for country in countries:
        dic1[country]=df[df['LCB ']==country]
        dic1[country].Label=dic1[country].Label.replace(np.nan,'normal')
    
    country = select

    if test_stationarity(dic[country], 'y')=='Stationary':
        TP,TN,FP,FN=[],[],[],[]
        for i in range(len(result)):
            if result['anomaly'].iloc[i]=='Yes' and dic1[country]['Label'].iloc[i]!='normal':
                TP.append(1)
            elif result['anomaly'].iloc[i]=='Yes' and dic1[country]['Label'].iloc[i]=='normal':
                FP.append(1)
            elif result['anomaly'].iloc[i]=='No' and dic1[country]['Label'].iloc[i]=='normal':
                TN.append(1)
            else:
                FN.append(1)

        acc=(len(TP)+len(TN))/(len(TP)+len(TN)+len(FP)+len(FN))
        precision=(len(TP))/(+len(TP)+len(FP))
        recall=len(TP)/(len(TP)+len(FN))
        F1_score=(2*precision*recall)/(precision+recall)
        false_positive_rate=len(FP)/(len(TP)+len(FP))
        missed=len(FN)/(len(FN)+len(TP))
    else:
        acc,precision,recall,F1_score,false_positive_rate,missed=0,0,0,0,0,0

    evaluation[country]={'Accuracy':acc,'Precision':precision,'Recall':recall,'F1_score':F1_score,'false_positive_rate':false_positive_rate,'missed_anomaly':missed}
    #print("accuracy: ",acc, ', Precision: ',precision," ,Recall: ",recall," ,F1_score: ",F1_score)
    evaluation_df=pd.DataFrame(evaluation)
    return evaluation_df


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

        # download as PDF
        st. markdown("### **Save to pdf**")
        pdf = FPDF('P', 'mm', 'A4')
        pdf.add_page()
        pdf.set_font(family='Arial', size=16)
        pdf.cell(40, 50, txt="Anomaly Detection Report")
        # pdf.cell(40, 50, txt=f"Overall {df_type} data in {select} from October 2020 to last week")
        pdf.image("fig1.png", w=195, h=65, y=40, x=10)

        st.sidebar.title("3. Model")
        model_option = st.sidebar.selectbox("Choose a model", ("ARIMA", "Coming soon"))

        if model_option == "ARIMA":
            m_path = os.path.join("models", "arima_model.json")

            st.markdown("## **Model prediction**")
            st.markdown(f"### Predicted anomalies in {df_type} data in {select} from October 2020 to last week")

            dic, pred, result = predict_model(m_path, df, select)
            evaluation_df = analyze_data(df, dic, select, pred, result)

            st.dataframe(evaluation_df, use_container_width=True)

            pdf.image("fig2.png", w=195, h=65, y=105, x=10)
        
        # download
        st.sidebar.download_button('Download report as PDF',
                        data=pdf.output(dest="S").encode("latin-1"),
                        file_name='anomaly_detection_report.pdf'
                        )

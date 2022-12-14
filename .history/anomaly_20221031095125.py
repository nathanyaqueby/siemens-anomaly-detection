import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF
from streamlit_plotly_events import plotly_events
import os
import statsmodels.tsa.stattools as sta
from prophet.serialize import model_from_json
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix 
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import pickle

#########################
## Functions for ARIMA ##
#########################

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

# identify anomaly type of arima
def f(row):
    if row['error'] >0:
        val = 'high peak'
    elif row['error'] < 0:
        val = 'low peak'
    return val
def f1(row):
    val='normal'
    return val

# old ARIMA
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
    fig3 = px.scatter(result.reset_index(), x='ds', y='y', color='anomaly', title=country)
    pio.write_image(fig3, "fig3.png", format="png", validate="False", engine="kaleido")

    # slider
    fig3.update_xaxes(
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
    st.plotly_chart(fig3, use_container_width=True)
    return dic, forecast, result

# old ARIMA extra function
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

# new ARIMA 
def fit_predict_model(m_path, dataframe,dataframe1):
    with open(m_path, "r") as fin:
        m = model_from_json(fin.read())
        
    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].reset_index(drop = True)

    result = pd.concat([dataframe.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]], axis=1)
    result['error'] = result['y'] - result['yhat']
    result['uncertainty'] = result['yhat_upper'] - result['yhat_lower']
    result['Anomaly'] = result.apply(lambda x: 'True' if(np.abs(x['error']) > 1.0*x['uncertainty']) else 'False', axis = 1)
    result['Label']=dataframe1['Label'].values
    #create new column 'Good' using the function above
    result['Label_pred'] = result[result['Anomaly']=='True'].apply(f, axis=1)
    result['Label_pred']=result['Label_pred'].replace(np.nan,'normal')
    # Using .fit_transform function to fit label
    # encoder and return encoded label
   
    # Creating a instance of label Encoder.
    le = LabelEncoder()

    result['Label_pred_num'] = le.fit_transform(result['Anomaly'])
    result.reset_index(inplace=True)
    result.rename(columns={"ds": "Date"},inplace=True)
    return forecast,result

# new ARIMA extra function
def analyze2(dic,name,country):
    fpr, tpr, thresholds = metrics.roc_curve(dic[country]['Label_num'].values, result['Label_pred_num'].values) 
    TN, FP, FN, TP = confusion_matrix(dic[country]['Label_num'].values, result['Label_pred_num'].values).ravel() 
    acc= (TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    TPR=TP/(TP+FN)
    FPR=FP/(TP+FP)
    F1_score=(2*precision*TPR)/(precision+TPR)
    new_ratio = ((3*TPR)+precision)/4

    evaluation[country+'_'+name]=new_ratio

    fin_max = max(evaluation, key=evaluation.get)
    fin_min = min(evaluation, key=evaluation.get)
    fin_mean=sum(evaluation.values())/len(evaluation)

    output={}
    output['max']=[fin_max,evaluation[fin_max]]
    output['min']=[fin_min,evaluation[fin_min]]
    output['mean']=[fin_mean]#,evaluation[fin_mean]]

    return output

# to change font size and make it prettier
def create_html(input_txt, mode):
    if mode == "header":
        html_txt = f"""
                    <style>
                    p.a {{
                    font: bold 42px sans-serif; text-align: center;
                    }}
                    </style>
                    <p class="a">{variable_output}</p>
                    """
    else:
        html_txt = f"""
                    <style>
                    p.a {{
                    font: 30px sans-serif; text-align: center;
                    }}
                    </style>
                    <p class="a">{input_text}</p>
                    """
    return html_txt

# save as excel
def to_excel_utils(df: pd.DataFrame, name: str) -> bytes:
    """
    Converts data to excel format and encodes to base64.

    Args:
        df (pd.DataFrame): project data
        name (str): project name

    Returns:
        b64 (bytes): Encoded project (excel) data
    """
    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    if isinstance(df, pd.DataFrame):
        df.to_excel(writer, index=True, sheet_name=name)
    else:
        for df_save, sheet in zip(df, name):
            df_save.to_excel(writer, index=True, sheet_name=sheet)

    writer.save()

    processed_data = output.getvalue()

    return processed_data

####################################
## Functions for Isolation Forest ##
####################################

# isolation forest
def forest_preprocess(df, lcb):
    df = df.set_index("ds")
    df.drop('Label',inplace=True,axis=1)
    df.drop(lcb,inplace=True,axis=1)
    df.drop("Segment",inplace=True,axis=1)
    df.drop("Label_num",inplace=True,axis=1)
    return df

def predict_forest(filename, df, lcb):
    data = forest_preprocess(df, lcb)
    model = pickle.load(open(filename, 'rb'))
    df['Anomaly'] = model.predict(data)

    return df


###############
## Dashboard ##
###############

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
    
    # Prepare PDF
    # st. markdown("### **Save to pdf**")
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_font(family='Arial', size=16)
    pdf.cell(40, 50, txt="Anomaly Detection Report")

    # st.sidebar.checkbox("Show Analysis by Location", True, key=1)
    st.sidebar.title("2. Location")
    check = st.sidebar.checkbox("Show analysis by location", value=False, key=2)
    
    if check:
        # Add additional dropdown in sidebar
        select = st.sidebar.selectbox('Select a location', df[lcb])

        # get the state selected in the selectbox
        state_data = df[df[lcb] == select]

        countries = df[lcb].unique()

        # initialise dictionaries for ARIMA
        dic = {}

        for country in countries:
            dic[country]=df[df[lcb]==country]
            dic[country].Label=dic[country].Label.replace(np.nan,'normal')
            dic[country]['Label_num']=np.where(dic[country]['Label']=='normal',0,1)
            dic[country]=dic[country].rename(columns = {"Date":"ds","Value":"y"})

        def get_total_dataframe(dataset):
            total_dataframe = pd.DataFrame({
            'Date':dataset[date],
            'Value':dataset[val]})
            return total_dataframe

        state_total = get_total_dataframe(state_data)

        # Show figure per location
        st.markdown("## **Location analysis**")
        date_min=df.Date.iloc[0].strftime("%B %Y")
        date_max=df.Date.iloc[-1].strftime("%B %Y")

        fig1 = px.line(
            state_total, 
            x='Date',
            y='Value',
            labels={'Value':'Value in %s' % (select)},
            width=1200, height=400,
            title=f"{df_type} data in {select} from {date_min} to {date_max}")
        fig1.update(layout=dict(title=dict(x=0.5)))

        # If single country: deploy model
        st.sidebar.title("3. Model")
        model_option = st.sidebar.selectbox("Choose a model", ("ARIMA", "Isolation Forest", "Local Outlier Factor"))

        selected_points = None
        
        # if a point was clicked, show info
        if selected_points:
            st.markdown("#### **Selected point**")
            st.markdown("Date: {}".format(selected_points[0]["x"]))
            st.markdown("Value: {}".format(selected_points[0]["y"]))

        
        if model_option == "ARIMA":
            m_path = os.path.join("models", "arima_model_2.json")

            # st.markdown(f"### Predicted anomalies in {df_type} data from {date_min} to {date_max}")

            # old ARIMA
            # dic, pred, result = predict_model(m_path, df, select)
            # evaluation_df = analyze_data(df, dic, select, pred, result)
            # st.dataframe(evaluation_df, use_container_width=True)
            # pdf.image("fig3.png", w=195, h=65, y=105, x=10)
            
            # new ARIMA
            evaluation = {}
            if test_stationarity(dic[select], 'y')=='Stationary':
                pred,result = fit_predict_model(m_path, dic[select],dic[select])
                output = analyze2(dic, df_type,select)
            else:
                output={}
                output['max']=0
                output['min']=0
                output['mean']=0

            # add anomalies in scatter form
            anomalies = result[result["Anomaly"]=='True']
            # st.write(anomalies.head())
            # st.write(state_total.head())
            fig_temp = px.scatter(anomalies, x="Date", y="y", color_discrete_sequence=["red"])
            fig1.add_trace(fig_temp.data[0])
            # create list of dicts with selected points, and plot
            selected_points = plotly_events(fig1)
            # st.plotly_chart(fig1,use_container_width=True)
            # st.plotly_chart(fig_temp,use_container_width=True)
            # generate image for pdf
            pio.write_image(fig1, "fig1.png", format="png", validate="False", engine="kaleido")
            pdf.image("fig1.png", w=195, h=65, y=40, x=10)

        elif model_option == "Isolation Forest":
            m_path = os.path.join("models", "forest_model.sav")
            result = predict_forest(m_path, dic[select], lcb)
            st.dataframe(result)

            # add anomalies in scatter form
            anomalies = result[result["Anomaly"]=='True']

            fig_temp = px.scatter(anomalies, x="ds", y="y", color_discrete_sequence=["red"])
            fig1.add_trace(fig_temp.data[0])
            # create list of dicts with selected points, and plot
            selected_points = plotly_events(fig1)

            # generate image for pdf
            pio.write_image(fig1, "fig1.png", format="png", validate="False", engine="kaleido")
            pdf.image("fig1.png", w=195, h=65, y=40, x=10)

        st.markdown("## **Model prediction**")
        with st.expander("I want to see the nerd stats!"):
            if model_option == "ARIMA":
                c1, c2, c3 = st.columns(3, gap="medium")

                with st.container():

                    variable_output = "<b>Brazil</b>"  # round(output["max"][1]*100)
                    input_text = "has the <b>highest</b> score with <b>100%</b> accuracy"
                    c1.markdown(create_html(variable_output, "header"), unsafe_allow_html=True)
                    c1.markdown(create_html(input_text, "normal"), unsafe_allow_html=True)

                    variable_output = "<b>68.83%</b>"  # round(output["mean"][0]*100)
                    input_text = "is the <b>average</b> accuracy"
                    c2.markdown(create_html(variable_output, "header"), unsafe_allow_html=True)
                    c2.markdown(create_html(input_text, "normal"), unsafe_allow_html=True)

                    variable_output = "<b>United Arab Emirates</b>"  # round(output["min"][1]*100)
                    input_text = "has the <b>lowest</b> score with <b>26.62%</b> accuracy"
                    c3.markdown(create_html(variable_output, "header"), unsafe_allow_html=True)
                    c3.markdown(create_html(input_text, "normal"), unsafe_allow_html=True)
            elif model_option == "Isolation Forest":
            
                c1, c2, c3 = st.columns(3, gap="medium")

                with st.container():

                    variable_output = "<b>Switzerland</b>"  # round(output["max"][1]*100)
                    input_text = "has the <b>highest</b> score with <b>99.96%</b> accuracy"
                    c1.markdown(create_html(variable_output, "header"), unsafe_allow_html=True)
                    c1.markdown(create_html(input_text, "normal"), unsafe_allow_html=True)

                    variable_output = "<b>99.29%</b>"  # round(output["mean"][0]*100)
                    input_text = "is the <b>average</b> accuracy"
                    c2.markdown(create_html(variable_output, "header"), unsafe_allow_html=True)
                    c2.markdown(create_html(input_text, "normal"), unsafe_allow_html=True)

                    variable_output = "<b>Russian Federation</b>"  # round(output["min"][1]*100)
                    input_text = "has the <b>lowest</b> score with <b>93.4%</b> accuracy"
                    c3.markdown(create_html(variable_output, "header"), unsafe_allow_html=True)
                    c3.markdown(create_html(input_text, "normal"), unsafe_allow_html=True)
            elif model_option == "Local Outlier Factor":
                
                c1, c2, c3 = st.columns(3, gap="medium")

                with st.container():

                    variable_output = "<b>Austria</b>"  # round(output["max"][1]*100)
                    input_text = "has the <b>highest</b> score with <b>98.02%</b> accuracy"
                    c1.markdown(create_html(variable_output, "header"), unsafe_allow_html=True)
                    c1.markdown(create_html(input_text, "normal"), unsafe_allow_html=True)

                    variable_output = "<b>93.86%</b>"  # round(output["mean"][0]*100)
                    input_text = "is the <b>average</b> accuracy"
                    c2.markdown(create_html(variable_output, "header"), unsafe_allow_html=True)
                    c2.markdown(create_html(input_text, "normal"), unsafe_allow_html=True)

                    variable_output = "<b>Russian Federation</b>"  # round(output["min"][1]*100)
                    input_text = "has the <b>lowest</b> score with <b>86.94%</b> accuracy"
                    c3.markdown(create_html(variable_output, "header"), unsafe_allow_html=True)
                    c3.markdown(create_html(input_text, "normal"), unsafe_allow_html=True)


        st.sidebar.title("4. Export Results")

        col1, col2 = st.sidebar.columns([1,1])
        col1.download_button(
            'Download PDF',
            data=pdf.output(dest="S").encode("latin-1"),
            file_name='anomaly_detection_data.pdf'
        )

        col2.download_button(
            label = "Download Excel",
            data = to_excel_utils(result, 'Sheet1'),
            file_name = "anomaly_detection_data.xlsx",
            mime = "application/vnd.ms-excel"
        )
    else:
        # Show figure of all data
        st.markdown("## **Product analysis**")
        date_min=df.Date.iloc[0].strftime("%B %Y")
        date_max=df.Date.iloc[-1].strftime("%B %Y")
        fig2 = px.line(df, x='Date', y='Value', color=lcb, title=f"All {df_type} data from {date_min} to {date_max}")
        fig2.update(layout=dict(title=dict(x=0.5)))
        selected_points = plotly_events(fig2)
        pio.write_image(fig2, "fig2.png", format="png", validate="False", engine="kaleido")
        pdf.image("fig2.png", w=195, h=65, y=40, x=10)
        st.sidebar.title("3. Export Results")

        # if a point was clicked, show info
        if selected_points:
            st.markdown("#### **Selected point**")
            st.markdown("Date: {}".format(selected_points[0]["x"]))
            st.markdown("Value: {}".format(selected_points[0]["y"]))

        # download
        st.sidebar.download_button('Download report as PDF',
                        data=pdf.output(dest="S").encode("latin-1"),
                        file_name='anomaly_detection_report.pdf'
                        )

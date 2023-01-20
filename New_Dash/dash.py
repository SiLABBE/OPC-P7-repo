import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'customer_data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def customer_data(data_path):
    df=pd.read_csv(data_path, nrows=50)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    customer_list = df["SK_ID_CURR"].drop_duplicates().to_list()
    return df, customer_list

def gauge(score, min=0, max=100):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted score"}))
    st.plotly_chart(fig, theme="streamlit")


def main():
    FastAPI_URI = 'http://127.0.0.1:8000/predict'
    data_path = 'df_train_filtered_50.csv'

    df_customer, customer_list = customer_data(data_path=data_path)

    st.title('Simulation for a customer loan request')
    selected_customer = st.text_input('Customer ID (format exemple : 200605):')
    customer_btn = st.button('Search for customer')

    if customer_btn:
        if int(selected_customer) in customer_list:
            filtered_customer = int(selected_customer)
            st.success("Selected customer : %s" %filtered_customer)
            df_filtered = df_customer[df_customer["SK_ID_CURR"]==filtered_customer]
            st.dataframe(df_filtered)

            X_cust = [i for i in df_filtered.iloc[:,2:].values.tolist()[0]]
            pred = request_prediction(FastAPI_URI, X_cust)
            st.write(
            'Predicted score: {}'.format(pred[0]))
            gauge(pred[0])

        else :
            st.warning("Unknown customer")

if __name__ == '__main__':
    main()

# streamlit run dash.py
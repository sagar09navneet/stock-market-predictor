# import numpy as np
# import pandas as pd
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# def prepare_data(df,forecast_col,forecast_out,test_size):
#     label = df[forecast_col].shift(-forecast_out) 
#     X = np.array(df[[forecast_col]]) 
#     X = preprocessing.scale(X) 
#     X_lately = X[-forecast_out:]
#     X = X[:-forecast_out] 
#     label.dropna(inplace=True) 
#     y = np.array(label)  
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0) #cross validation

#     response = [X_train,X_test , Y_train, Y_test , X_lately]
#     return response
# data = {
#     'Date': pd.date_range(start='1/1/2022', periods=10),
#     'Price': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
# }
# df = pd.DataFrame(data)
# forecast_col='Price'
# forecast_out=2
# test_size=0.2

# X_train, X_test, Y_train, Y_test , X_lately =prepare_data(df,forecast_col,forecast_out,test_size); 
# learner = LinearRegression() 

# learner.fit(X_train,Y_train)

# score=learner.score(X_test,Y_test)
# forecast= learner.predict(X_lately) 
# response={}
# response['test_score']=score
# response['forecast_set']=forecast

# print(response)

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out)
    X = np.array(df[[forecast_col]])
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    label.dropna(inplace=True)
    y = np.array(label)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response


st.title("Stock Price Predictor")

st.write("""
This app predicts the future stock prices based on the provided dataset.
""")


uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    forecast_col = st.selectbox("Select column to forecast", df.columns)
    forecast_out = st.number_input("Enter number of days to forecast out", min_value=1, max_value=len(df) - 1, value=2)
    test_size = st.slider("Select test size (as a fraction of the dataset)", min_value=0.1, max_value=0.9, value=0.2)

    if st.button("Predict"):
        X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)

        learner = LinearRegression()
        learner.fit(X_train, Y_train)
        score = learner.score(X_test, Y_test)
        forecast = learner.predict(X_lately)

        response = {}
        response['test_score'] = score
        response['forecast_set'] = forecast

        st.write("Test Score (R^2):", response['test_score'])
        st.write("Forecast Set:", response['forecast_set'])
else:
    st.info("Please upload a CSV file to get started.")



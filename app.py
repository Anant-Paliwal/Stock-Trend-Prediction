import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import streamlit as st

from keras.models import load_model

from pandas_datareader import data as data
import yfinance as yfin
msft = yfin.Ticker("MSFT")

start="2010-01-01"
end="2022-12-31"

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yfin.download(user_input,start,end )

st.subheader('Date from 2010 - 2022')
st.write(df.describe())

# visulization

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA to 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# Traning and Testing data

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

# Splitting Data into x_train and y_train


model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, data_training_array.shape[0]):
    x_test.append(data_training_array[i-100 : i])
    y_test.append(data_training_array[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scaler_factor = 1/scaler[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

def graph():
    st.subheader('Prediction vs Original')
fig2 =plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Prediction Price')
plt.xlabel('Time')
plt.xlabel('Price')
plt.legend()
st.pyplot(fig2)

def graph1():
    st.subheader('Prediction vs Original')
fig2 =plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Prediction Price')
plt.xlabel('Time')
plt.xlabel('Price')
plt.legend()
st.pyplot(fig2)

# option = st.selectbox(
#      'How would you like to be contacted?',
#      (graph(fig2), graph1(fig2)))
# st.write(option)

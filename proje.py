import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_option('deprecation.showfileUploaderEncoding', False)

model = load_model(r'C:\Users\naife\OneDrive\Masaüstü\VM_proje\Stock_Predictions_Model.keras')

st.header('Borsa Tahmin Uygulaması')

stock = st.text_input('Hisse sembolü', 'AAPL')

aapl = pd.read_csv(r'C:\Users\naife\OneDrive\Masaüstü\AAPL(80-24).CSV')

st.subheader('Günlük hisse fiyatı')
st.write(aapl)

aapl_train = pd.DataFrame(aapl.Open[0: int(len(aapl)*0.80)])
aapl_test = pd.DataFrame(aapl.Open[int(len(aapl)*0.80):len(aapl)])

scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = aapl_train.tail(100)
aapl_test = pd.concat([pas_100_days, aapl_test], ignore_index=True)
aapl_test_scale = scaler.fit_transform(aapl_test)

st.subheader('Fiyat vs MA50')
ma_50_days = aapl.Open.rolling(50).mean()

fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=aapl.index, y=ma_50_days, mode='lines', name='MA50'))
fig1.add_trace(go.Scatter(x=aapl.index, y=aapl.Open, mode='lines', name='Fiyat'))

fig1.update_layout(title='Fiyat vs MA50', xaxis_title='Tarih', yaxis_title='Fiyat')
st.plotly_chart(fig1)

st.subheader('Fiyat vs MA50 vs MA100')
ma_100_days = aapl.Open.rolling(100).mean()

fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=aapl.index, y=ma_50_days, mode='lines', name='MA50'))
fig2.add_trace(go.Scatter(x=aapl.index, y=ma_100_days, mode='lines', name='MA100'))
fig2.add_trace(go.Scatter(x=aapl.index, y=aapl.Open, mode='lines', name='Fiyat'))

fig2.update_layout(title='Fiyat vs MA50 vs MA100', xaxis_title='Tarih', yaxis_title='Fiyat')
st.plotly_chart(fig2)

st.subheader('Fiyat vs MA100 vs MA300')
ma_300_days = aapl.Open.rolling(300).mean()

fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=aapl.index, y=ma_100_days, mode='lines', name='MA100'))
fig3.add_trace(go.Scatter(x=aapl.index, y=ma_300_days, mode='lines', name='MA300'))
fig3.add_trace(go.Scatter(x=aapl.index, y=aapl.Open, mode='lines', name='Fiyat'))

fig3.update_layout(title='Fiyat vs MA100 vs MA300', xaxis_title='Tarih', yaxis_title='Fiyat')
st.plotly_chart(fig3)

x = []
y = []

for i in range(100, aapl_test_scale.shape[0]):
    x.append(aapl_test_scale[i-100:i])
    y.append(aapl_test_scale[i,0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Gerçek Fiyat vs Modelin Tahmini')

fig4 = go.Figure()

fig4.add_trace(go.Scatter(x=aapl.index[100:], y=predict.flatten(), mode='lines', name='Tahmin Fiyat', line=dict(color='red')))
fig4.add_trace(go.Scatter(x=aapl.index[100:], y=y, mode='lines', name='Gerçek Fiyat', line=dict(color='green')))

fig4.update_layout(title='Gerçek Fiyat vs Modelin Tahmini', xaxis_title='Tarih', yaxis_title='Fiyat')
st.plotly_chart(fig4)

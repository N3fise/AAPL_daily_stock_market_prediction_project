# Zaman serisi yöntemini ekleyerek kodunuzu düzenleyin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

# Veriyi pandas ile alıyorum.
aapl = pd.read_csv(r'C:\Users\naife\OneDrive\Masaüstü\AAPL(80-24).CSV', delimiter=',')
print(aapl.head())

# Eksik veri var mı kontrol ediyorum.
eksik_degerler = aapl.isnull()
print("Eksik Değerler:\n", eksik_degerler)

# Ortalama almak için 2000 yılının verilerini buldum
secilen_veriler = aapl[aapl['Date'].str.contains('/2000')]

# Seçilen verileri görüntüleyin
print(secilen_veriler)

# secilen_veriler DataFrame'inizdeki 'Price' sütununun ortalamasını alın
ortalama_price = secilen_veriler['Price'].mean()
ortalama_open = secilen_veriler['Open'].mean()
ortalama_high = secilen_veriler['High'].mean()
ortalama_low = secilen_veriler['Low'].mean()

# Ortalama fiyatı yazdırın
print("Ortalama Price:", ortalama_price)
print("Ortalama open:", ortalama_open)
print("Ortalama high:", ortalama_high)
print("Ortalama low:", ortalama_low)

# Dizini sıfırladım, yani index değerleri atadım.
aapl.reset_index(inplace=True)
print(aapl)

# 100 günlük hareketli ortalamanın grafiğini çizdirdim.
ma_500_days = aapl.Open.rolling(500).mean()
plt.figure(figsize=(8,6))
plt.plot(ma_500_days, 'r')
plt.plot(aapl.Open, 'g')
plt.show()

# Son 200 günün hareketli ortalamasının grafiğini çizdiridm.
ma__1000_days = aapl.Open.rolling(1000).mean()
plt.figure(figsize=(8,6))
plt.plot(ma_500_days, 'r')
plt.plot(ma__1000_days, 'b')
plt.plot(aapl.Open, 'g')
plt.show()

# Verilerimi eğitim ve test verisi olarak ikiye ayırdım
aapl_train = pd.DataFrame(aapl.Open[0: int(len(aapl)*0.80)])
aapl_test = pd.DataFrame(aapl.Open[int(len(aapl)*0.80):len(aapl)])

# Verilerimi (0,1) aralığında ölçeklendirdim. Min-max normalizasyon kullandım.
scaler = MinMaxScaler(feature_range=(0,1))
aapl_train_scale = scaler.fit_transform(aapl_train)

# Zaman serisi verilerinin sıralı bir şekilde işlenmesinde kullanılır.
x = []
y = []

for i in range(100, aapl_train_scale.shape[0]):
    x.append(aapl_train_scale[i-100:i, 0])  # Önceki 100 günlük veri dilimi
    y.append(aapl_train_scale[i, 0])         # Sonraki günün verisi
x, y = np.array(x), np.array(y)

# Verileri yeniden şekillendirme
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Model oluşturma
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120))
model.add(Dropout(0.5))
model.add(Dense(units=1))

# Modeli derleme
model.compile(optimizer='adam', loss='mse')

# Modeli eğitme
model.fit(x, y, epochs=30, batch_size=35, verbose=1)

# Modelin özetini görüntüleme
model.summary()

# Test verisi ile eğitim verisini birleştirme
aapl_test = pd.concat([aapl_train.tail(500), aapl_test], ignore_index=True)

# Test verisini ölçeklendirme
aapl_test_scale = scaler.fit_transform(aapl_test)

# Test verisi için x ve y oluşturma
x_test = []
y_test = []

for i in range(100, aapl_test_scale.shape[0]):
    x_test.append(aapl_test_scale[i-100:i, 0])
    y_test.append(aapl_test_scale[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Verileri yeniden şekillendirme
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Tahmin yapma
y_predict = model.predict(x_test)

# Tahminleri ters ölçekleme
y_predict = scaler.inverse_transform(y_predict)
y_test = scaler.inverse_transform([y_test])

# Tahminleri ve gerçek değerleri görselleştirme
plt.figure(figsize=(8, 6))
plt.plot(y_predict, 'r', label='Tahmin Edilen Fiyat')
plt.plot(y_test[0], 'g', label='Gerçek Fiyat')
plt.xlabel('Zaman')
plt.ylabel('Fiyat')
plt.legend()
plt.show()

# Hata hesaplama
mse = mean_squared_error(y_test[0], y_predict)
print("Mean Squared Error (MSE):", mse)

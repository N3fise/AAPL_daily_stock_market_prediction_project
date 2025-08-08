import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

#veriyi pandas ile alıyorum.
aapl = pd.read_csv(r'C:\Users\naife\OneDrive\Masaüstü\AAPL(80-24).CSV', delimiter=',')
print(aapl.head())

#dizini sıfırladım
aapl.reset_index(inplace=True)
print(aapl)


 # ön işleme.
eksik_degerler = aapl[aapl.isnull().any(axis=1)]
print("Eksik Değerler:\n", eksik_degerler)

#ortalama almak için 2000 yılının verilerini buldum
secilen_veriler = aapl[aapl['Date'].str.contains('/2000')]

# Seçilen verileri görüntüleyin
print(secilen_veriler)

# secilen_veriler DataFrame'inizdeki 'Price' sütununun ortalamasını alın
ortalama_price = secilen_veriler['Price'].mean()
ortalama_open = secilen_veriler['Open'].mean()
ortalama_high = secilen_veriler['High'].mean()
ortalama_low = secilen_veriler['Low'].mean()
#ortalama_vol = secilen_veriler['Vol.'].mean()
#ortalama_change = secilen_veriler['Change %'].mean()

# Ortalama fiyatı yazdırın
print("Ortalama Price:", ortalama_price)
print("Ortalama open:", ortalama_open)
print("Ortalama high:", ortalama_high)
print("Ortalama low:", ortalama_low)
#print("Ortalama Price:", ortalama_vol) 
#print("Ortalama Price:", ortalama_chang)"""
"""vol. ve change verilerini int olmadığı için onlarla işlem yapamadım. manuel olarak doldurcağım."""

#ön işlemeyi bitirdim kontrol ediyorum.
eksik_degerler = aapl.isnull().sum()
print("Eksik Değerler:\n", eksik_degerler)

# artık eksik değer yok. modeli eğitmeye geçiyorum.

#verielrimi eğitim ve test verisi olarak ikiye ayırdım
aapl_train = pd.DataFrame(aapl.Open[0: int(len(aapl)*0.80)])
aapl_test = pd.DataFrame(aapl.Open[int(len(aapl)*0.80):len(aapl)])
 
#eğitim
print(aapl_train.shape[0])

#test
print(aapl_test.shape[0])

#verilerimi (0,1) aralığında ölçeklendirdim.
scaler = MinMaxScaler(feature_range=(0,1))
aapl_train_scale = scaler.fit_transform(aapl_train)

#zaman serisi verilerinin sıralı bir şekilde işlenmesinde kullanılır.
x = []
y = []
 
for i in range(100, aapl_train_scale.shape[0]):
    x.append(aapl_train_scale[i-100:i])
    y.append(aapl_train_scale[i,0])
x,y = np.array(x), np.array(y)    

#model
model = Sequential()
model.add(LSTM(units = 50, activation='relu', return_sequences = True, input_shape = ((x.shape[1],1)))) 
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation='relu', return_sequences = True))   
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation='relu', return_sequences = True))   
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation='relu'))   
model.add(Dropout(0.5))

model.add(Dense(units = 1))

#modeli derliyorum.
model.compile(optimizer = 'adam', loss = 'mse')
#50 epoch 
model.fit(x,y, epochs = 3, batch_size = 25, verbose = 1)
model.summary()

# son 500 günlük veriyi seçerek yeni bir data frame oluşturfum.
pas_500_days = aapl_train.tail(500)

# test verisi ile eğitim verisini karşılaştırıyom.
aapl_test = pd.concat([pas_500_days, aapl_test], ignore_index = True)


aapl_test_scale = scaler.fit_transform(aapl_test)
x = []
y = []
 
for i in range(500, aapl_test_scale.shape[0]):
    x.append(aapl_test_scale[i-500:i])
    y.append(aapl_test_scale[i,0])
x,y = np.array(x), np.array(y)  
 
# x giriş verisinden tahmin yaparak y_predict'e atadım.
y_predict = model.predict(x)

# tahminler 
print(y_predict)

#dönüşüm
scale = scaler.scale_
print(scale)

# tahmini de dönüştürüyorum
y_predict = y_predict*scale

y = y*scale

plt.figure(figsize=(8,10))
plt.plot(y_predict, 'r', label = 'predict Price')
plt.plot(y, 'g', label = 'original Price')
plt.xlabel('zaman')
plt.ylabel('Fiyat')
plt.legend
plt.show() 

























from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM


# Veri yükleme
Data = pd.read_csv(r'C:\Users\naife\OneDrive\Masaüstü\DE.csv')
print(Data.head())

Data.reset_index(inplace=True)
print(Data)

#verielrimi eğitim ve test verisi olarak ikiye ayırdım
Data_train = pd.DataFrame(Data.Open[0: int(len(Data)*0.80)])
Data_test = pd.DataFrame(Data.Open[int(len(Data)*0.80):len(Data)])

#eğitim
print(Data_train.shape[0])
#test
print(Data_test.shape[0])

#verilerimi (0,1) aralığında ölçeklendirdim.
scaler = MinMaxScaler (feature_range=(0,1))
Data_train_scale = scaler.fit_transform(Data_train)

x = []
y = []
 
for i in range(100, Data_train_scale.shape[0]):
    x.append(Data_train_scale[i-100:i])
    y.append(Data_train_scale[i, 0]) 
x, y = np.array(x), np.array(y)
  

#model
model = Sequential()
model.add(LSTM(units = 10, activation='relu', return_sequences = True, input_shape = ((x.shape[1],1)))) 
model.add(Dropout(0.2))
model.add(LSTM(units = 20, activation='relu'))   
model.add(Dropout(0.3))

#modeli derliyorum.
model.compile(optimizer = 'adam', loss = 'mse')

#50 epoch 
model.fit(x,y, epochs = 3, batch_size = 3, verbose = 1)
model.summary()



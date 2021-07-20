#RECURRENT NEURAL NETWORK

#DATA PREPROCESSING

#IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#IMPORTING THE TRAINING SET
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#FEATURE SCALING
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#CREATING A DATA STRUCTURE WITH 60 TIMESTEPS AND OUTPUT
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#RESHAPING
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#BUILDING THE RNN

#IMPORTING THE KERAS LIBRARIES AND PACKAGES
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#INITIALISING THE RNN
regressor = Sequential() 

#ADDING THE FIRST LSTM LAYER AND SOME DROPOUT REGULARISATION(TO AVOID OVERFITTING)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(rate = 0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(rate = 0.2))

#ADDING THE OUTPUT LAYER
regressor.add(Dense(units = 1))

#COMPILING THE RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#FITTING THE RNN TO THE TRAINING SET
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#SAVE FILE TO CURRENT WORKING DIRECTORY
RNN = "RNN.pkl"
with open(RNN, 'wb') as file:
    pickle.dump(regressor, file)
    
#LOAD FROM FILE
with open(RNN, 'rb') as file:
    RNN_regressor = pickle.load(file)

#MAKING THE PREDICTIONS AND VISUALISING THE RESULTS

#GETTING THE REAL STOCK PRICE OF 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#GETTING THE PREDICTED STOCK PRICE OF 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = RNN_regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#VISUALISING RESULTS
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Predictions')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#EVALUATING MODEL PERFORMANCE USING ROOT MEAN SQUARED
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
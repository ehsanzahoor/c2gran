import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pyramid.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA,ARIMAResults
from sklearn.metrics import mean_squared_error, r2_score
#import datapackage
import warnings
#warnings.filterwarnings('ignore')

#Function to take moving average of an array/series
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

data = pd.read_csv('server2_status_final.csv')
print(data.info())
parameter_2_train = 'total_calls'
# Computing and Adding dcec column
#data['dcec'] = data['total_calls']*1000/(data['CO2_UK']*(data['power watts']-311))
# 1000 is multiplied to change CO2e from g/kWh to g/Wh
# 311 is subtracted becasue 311 is the power consumption in idle state. We only need power consumed by the number of calls
data['dcec'] = data['total_calls']*1000/(data['CPU %']*data['RAM Change %']*data['jitter']*data['delay']*data['CO2_UK']*(data['power watts']-311))

#print(data.head())
#print(data)
data['time_now'] = pd.to_datetime(data['time_now'])
data.set_index('time_now',inplace=True,drop=True)
#print(data.head())

plt.figure(figsize=(20,6))
#plt.plot(data['total_calls'],label='# of VOIP streams')
#plt.plot(data['jitter'],label='Jitter')
#plt.plot(data['delay'],label='Delay')
#plt.plot(data['dcec'],label='DC Contrbuticity Value')
#plt.legend()
#plt.savefig('ml_plots_models/delay_jitter.png')

# Taking different options for trainign and testing data
#train, test = train_test_split(data, test_size=0.2)
#train = data.head(2000).append(data.tail(6881))
#test = data.iloc[2001:4000]
#train = data.iloc[-8881:]
#test = data.iloc[0:2000]
train = data.iloc[:-2000]
test = data.iloc[-2000:]

print("Train size: %d"%train.shape[0])
print("Test size: %d"%test.shape[0])

# ARIMA Model
# Uncomment following block/s if you need to use ARIMA model for training
'''
stepwise_fit = auto_arima(data['dcec'],start_p=0,start_q=0,max_p=5,max_q=3,seasonal=False)
print(stepwise_fit.summary())

model = ARIMA(train['dcec'],order=(1, 1, 2))
results = model.fit()
print(results.summary())

start = len(train)
end = len(train) + len(test) - 1

predictions = results.predict(start=start, end=end, typ='levels').rename('ARIMA (1,1,2) Predictions')

plt.figure(figsize=(20,6))
plt.plot(test['dcec'], label='Actual DCeC')
plt.plot(predictions, label='Predicted DCeC')
#test['dcec'].plot(figsize=(20,6),legend=True)
#predictions.plot(legend=True)
plt.legend()
plt.savefig('ml_plots_models/ARIMA_dcec.png')

print('MSE: %f'%mean_squared_error(test[parameter_2_train],predictions))
print('RMSE: %f'%np.sqrt(mean_squared_error(test[parameter_2_train],predictions)))
print('R-Squared: %f'%r2_score(test[parameter_2_train],predictions))
'''

#print(train[['CPU %']].head())
#train_array = train.to_numpy()#train[['total_calls'],['bytes_transmitted_kb'] ['CPU %'], ['Total Ram Occupied %'], ['power watts'], ['CO2_IE']]

sc = StandardScaler()
sc.fit(train[[parameter_2_train]])

scaled_train = sc.transform(train[[parameter_2_train]])
scaled_test = sc.transform(test[[parameter_2_train]])

X_train = scaled_train[:-1]
y_train = scaled_train[1:]

X_test = scaled_test[:-1]
y_test = scaled_test[1:]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print(X_train)
print(y_train)

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,LSTM
from keras.optimizers import Adam
import keras.backend as K

# FNN Here
# Uncomment following block/s if you need to use FFNN for training

'''
K.clear_session()
np.random.seed(7)
model = Sequential()

model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(1, activation='relu'))
#model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001))
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)
model.save('ml_plots_models/jitter_FNN_model.h5')
print('Model saved')

X_test_NN = np.insert(X_test,0,X_train[-1][0])
predictions = model.predict(X_test_NN)
plt.figure(figsize=(20,6))
plt.plot(y_test, label='Actual Jitter Values')
plt.plot(predictions, label='Predicted Jitter Values')
plt.legend()
plt.savefig('ml_plots_models/jitter_FNN.png')
predictions_inv_scaled = sc.inverse_transform(predictions)
y_test_inverse = sc.inverse_transform(y_test)
plt.figure(figsize=(20,6))
plt.plot(y_test_inverse, label='Actual Jitter Values')
plt.plot(predictions_inv_scaled, label='Predicted Jitter Values')
plt.legend()
plt.savefig('ml_plots_models/jitter_inversed_FNN.png')

#print(predictions)
#print(predictions_inv_scaled)
print('MSE: %f'%mean_squared_error(test[[parameter_2_train]],predictions_inv_scaled))
print('RMSE: %f'%np.sqrt(mean_squared_error(test[[parameter_2_train]],predictions_inv_scaled)))
print('R-Squared: %f'%r2_score(test[[parameter_2_train]],predictions_inv_scaled))
'''

#LSTM

X_train_t = X_train[:,None]
X_test_t = X_test[:,None]
print(X_train_t.shape)
print(X_test_t.shape)

print(X_train_t)
np.random.seed(7)

K.clear_session()

model = Sequential()

#Uncomment these lines if you want to train new LSTM model
'''
model.add(LSTM(20, input_shape=(1,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['mse', 'mae', 'mape', 'cosine', 'accuracy'])
history = model.fit(X_train_t, y_train, validation_data=(X_test_t, y_test), epochs=100, verbose=0)
model.save('ml_plots_models/dcec_new_s1_LSTM_model.h5')
print('Model saved')

plt.figure(figsize=(20,6))
plt.plot(history.history['mse'], label="Training Mean Squared Error")
plt.plot(history.history['mae'], label='Training Mean Absolute Error')
plt.plot(history.history['cosine'], label='Trainging Cosine Proximity')
plt.plot(history.history['val_mse'], label="Testing Mean Squared Error")
plt.plot(history.history['val_mae'], label='Testing Mean Absolute Error')
plt.plot(history.history['val_cosine'], label='Testing Cosine Proximity')
plt.legend()
plt.savefig('ml_plots_models/dcec_errors_s1_LSTM.png')
'''
# Load already trained LSTM model
model = load_model('ml_plots_models/dcec_LSTM_model.h5')
print('Model loaded')
print(X_train[-1][0])
X_test_LSTM = np.insert(X_test,0,X_train[-1][0]).reshape(-1,1)
X_test_LSTM = X_test_LSTM[:,None]
print(X_test_LSTM.shape)
predictions = model.predict(X_test_LSTM)
plt.figure(figsize=(20,6))
plt.plot(moving_average(y_test, 50), label='Actual number of streams')
plt.plot(moving_average(predictions, 50), label='Predicted number of streams')
plt.legend()
#plt.show()
plt.savefig('ml_plots_models/total_calls_s1_LSTM.png')
predictions_inv_scaled = sc.inverse_transform(predictions)
y_test_inverse = sc.inverse_transform(y_test)
print(predictions_inv_scaled)
plt.figure(figsize=(20,6))
plt.plot(moving_average(y_test_inverse,50), label='Actual number of streams')
plt.plot(moving_average(predictions_inv_scaled,50), label='Predicted number of streams')
plt.legend()
#plt.show()
plt.savefig('ml_plots_models/total_calls_inversed_s1_LSTM.png')
#print(predictions_inv_scaled.shape)
#print(X_test.shape)
#print(y_test.shape)
#df_pred = pd.DataFrame(predictions_inv_scaled)
#print(df_pred.shape)
#df_pred.to_csv('predictions.csv', index=False)
#df_test = pd.DataFrame(sc.inverse_transform(y_test))
#df_test.to_csv('y_test.csv',index=False)
#df_xtest = pd.DataFrame(sc.inverse_transform(X_test))
#df_xtest.to_csv('x_test.csv',index=False)
print('MSE: %f'%mean_squared_error(test[[parameter_2_train]],predictions_inv_scaled))
print('RMSE: %f'%np.sqrt(mean_squared_error(test[[parameter_2_train]],predictions_inv_scaled)))
print('R-Squared: %f'%r2_score(test[[parameter_2_train]],predictions_inv_scaled))

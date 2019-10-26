from __future__ import absolute_import, division, print_function, unicode_literals
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from pylab import *

def convert_raw_data_to_timeseries(scaled, n_steps_in, n_steps_out, cash_in):
    """
	:param data: DataFrame
	:param n_steps_in: Int
	:param n_steps_out: Int
	:param cash_in: Boolean
	:return: numpy.array, numpy.array
	"""
    X, y = list(), list()
    # normalize features

    for i in range(len(scaled)):

        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        if out_end_ix > len(scaled):
            break

        if cash_in:
            seq_x, seq_y = scaled[i:end_ix, :], scaled[end_ix:out_end_ix, 0:1]

        else:
            seq_x, seq_y = scaled[i:end_ix, :], scaled[end_ix:out_end_ix, 1:2]

        X.append(seq_x)
        y.append(seq_y)

    arrayX = np.array(X)
    arrayY = np.array(y)

    arrayY = arrayY.reshape(arrayY.shape[0], arrayY.shape[1])

    return arrayX, arrayY

# LSTM
def runModel(scaledArray, n_features, n_days_to_feed_model, n_days_to_predict, cash_in):
    """
    :param model: LSTM
	:param scaledArray: numpy.Array
	:param n_features: Int
	:param n_days_to_feed_model: Int
	:param n_days_to_predict: Int
	:param cash_in: Boolean
	:return: keras.callbacks.callbacks.History, test_X, test_y, prediction_values
	"""

    numberOfExamples = len(scaledArray)
    testDays = 28
    predictionDays = 14
    train = scaledArray[0: numberOfExamples - testDays , :]
    test = scaledArray[numberOfExamples - testDays: numberOfExamples, :]
    prediction_values = scaledArray[numberOfExamples - predictionDays: numberOfExamples, :]

    print("TRAIN LENGTH {}".format(len(train)))
    print("TEST LENGTH {}".format(len(test)))
    print("PREDICT LENGTH {}".format(len(prediction_values)))

    # Predict CashIn or CashOut
    train_X, train_y = convert_raw_data_to_timeseries(train, n_days_to_feed_model, n_days_to_predict, cash_in)
    test_X, test_y = convert_raw_data_to_timeseries(test, n_days_to_feed_model, n_days_to_predict, cash_in)

    # network
    model_ = Sequential()
    model_.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_days_to_feed_model, n_features)))
    model_.add(LSTM(100, activation='relu'))
    model_.add(Dense(n_days_to_predict))
    model_.compile(optimizer='adam', loss='mse')
    model_.summary()

    # fit network
    history = model_.fit(train_X, train_y, epochs=2, batch_size=72, validation_data=(test_X, test_y), verbose=1,
                        shuffle=False)
    return model_, history, test_X, test_y, prediction_values


def preprocessModelDf(modelDf, categorical_feature_index_list):
    """
	:param modelDf: DataFrame
	:param categorical_feature_index_list: List
	:return: scaled Numpy Array
	"""

    values = modelDf.values
    # integer encode direction

    if len(categorical_feature_index_list):
        encoder = LabelEncoder()
        for categorical_feature in categorical_feature_index_list:
            values[:, categorical_feature] = encoder.fit_transform(values[:, categorical_feature])

    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return  scaled


def plotLossHistory(history):
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

def makePrediction( test_X, test_y, prediction_values, model_):
    """
    :param scaler: Numpy scaler
	:param test_X: Numpy Array
	:param test_y: Numpy Array
	:param prediction_values: Numpy Array
	:param model: LSTM
	:return: rmse Double, predicted_values Numpy Array
	"""
    print("test X shape {}".format(test_X.shape))
    print("test y shape {}".format(test_y.shape))
    print("prediction values shape {}".format(prediction_values.shape))

    scaler_1= MinMaxScaler(feature_range=(0, 1))
    scaler_2 = MinMaxScaler(feature_range=(0, 1))
    scaler_3 = MinMaxScaler(feature_range=(0, 1))

    yhat = model_.predict(test_X, verbose=0)
    print("yhat shape {}".format(yhat.shape))

    prediction_values = prediction_values.reshape(1, prediction_values.shape[0], prediction_values.shape[1] )
    predicted_values = model_.predict(prediction_values, verbose=0)
    print("predicted_values values shape {}".format(predicted_values.shape))

    scaler_1 = scaler_1.fit(test_y)
    inv_y = scaler_1.inverse_transform(test_y)

    scaler_2 = scaler_2.fit(yhat)
    inv_yhat = scaler_2.inverse_transform(yhat)

    rmse = sqrt(mean_squared_error(inv_yhat, inv_y))
    print('Test RMSE: %.3f' % rmse)

    scaler_3 = scaler_3.fit(predicted_values)
    inv_predicted_values = scaler_3.inverse_transform(predicted_values)

    return rmse, inv_predicted_values

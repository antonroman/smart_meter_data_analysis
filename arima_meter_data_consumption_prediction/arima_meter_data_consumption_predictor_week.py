#!/usr/bin/python3
# train and test sets


# for f in ../data/datoscontadores_csv/S05/*.csv ; do ./arima_meter_data_consumption_predictor_week.py -i "$f" ;done


import sys, getopt
# arima forecast
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('arima_meter_data_consumption_predictor_week.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('arima_meter_data_consumption_predictor_week.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    return inputfile

# split a univariate dataset into train/test sets
def split_dataset(data):
    #print(len(data))
	# split into standard weeks
    train,test = data[70:364],data[0:70]
    # print(train.shape)
    # print(test.shape)
	# # restructure into windows of weekly data
    # print(len(train))
    # print(len(test))
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test
 
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# evaluate a single model
def evaluate_model(model_func, train, test):
	# history is a list of weekly data
    # print('TRAIN:')
    # print(train)
    history = [x for x in train]
    # print('HISTORY: ')
    # print(history)
	# walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = model_func(history)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = array(predictions)
    # evaluate predictions days for each week
    # the diff is stored in column 8
    score, scores = evaluate_forecasts(test[:, :, 8], predictions)
    return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
    # print('DATA:')
    # print(data)
    series = [week[:, 8] for week in data]
    # print('SERIES BEFORE FLATTENING:')
    # print(series)
    # flatten into a single series
    series = array(series).flatten()
    # print('SERIES AFTER FLATTENING:')
    # print(series)
    return series

# arima forecast
def arima_forecast(history):
	# convert history into a univariate series
    series = to_series(history)
    # print('SERIES: ')
    # print(type(series))
    # print(len(series))
    # define the model
    model = ARIMA(series, order=(7,0,0))
    # fit the model
    model_fit = model.fit(disp=False)
    # make forecast
    yhat = model_fit.predict(len(series), len(series)+6)
    # print('YHAT')
    # print(yhat)
    return yhat


if __name__ == "__main__":
    input_file = main(sys.argv[1:])
    print('Processing file '+ input_file + '...')
    # load the new file
    #dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    #samplesS05_df = pd.read_csv('meter_data_ZIV0035301588_S05.csv', header=0, infer_datetime_format=True, index_col=['Fh'])
    samplesS05_df = pd.read_csv(input_file,infer_datetime_format=True)
    #print(samplesS05_df.head())
    samplesS05_df['Fh'] = pd.to_datetime(samplesS05_df['Fh'])

    samplesS05_df= samplesS05_df.sort_values(by=['Fh'])
    samplesS05_df = samplesS05_df.set_index('Fh')
    samplesS05_df['AI-diff'] = samplesS05_df.diff(periods=1,axis=0)['AI-Total']
    #samplesS05_df['AI-diff'] = samplesS05_df.diff(periods=1,axis=0)['AI-Total']
    #samplesS05_df= samplesS05_df.sort_values(ascending=False,by=['Fh'])

    # weekly_consumption = samplesS05_df['AI-diff'].resample('W').sum()
    # print(weekly_consumption)
    # print(type(weekly_consumption))

    #print(samplesS05_df)
    #print(samplesS05_df.tail())

    #print(samplesS05_df.values)

    train, test = split_dataset(samplesS05_df.values)

    series = to_series(train)
    print(type(series))
    print(series)
    # plots
    pyplot.figure()
    lags = 365
    # acf
    axis = pyplot.subplot(2, 1, 1)
    plot_acf(series, ax=axis, lags=lags)
    # pacf
    axis = pyplot.subplot(2, 1, 2)
    plot_pacf(series, ax=axis, lags=lags)
    # show plot
    pyplot.show()
    # print('TRAIN::::')
    # print(train.shape)
    # print(train)
    # print('TEST:::')
    # print(test.shape)
    # print(test)
    # #train = split_dataset(dataset.values)
    # # validate train data
    # print('Tipo: ')
    # print(type(train))
    # print(train)
    # print(train.shape)
    #print(train[0, 0, 0], train[0,0,10], train[-1, -1, 0])
    # # # validate test
    # print(test.shape)
    # print(test[0, 0, 0], test[-1, -1, 0])
    #print(samplesS05_df.diff(periods=1,axis=0)['AI-Total'])

    # split into train and test
    #train, test = split_dataset(samplesS05_df.values)
    # define the names and functions for the models we wish to evaluate
    models = dict()
    models['arima'] = arima_forecast
    # evaluate each model
    days = ['mon', 'tue', 'wed', 'thr', 'fri', 'sat', 'sun']
    for name, func in models.items():
        # evaluate and get scores
        score, scores = evaluate_model(func, train, test)
        # summarize scores
        summarize_scores(name, score, scores)
        print('SCORES:')
        # plot scores
        print(scores)
    #   pyplot.plot(days, scores, marker='o', label=name, linestyle='None')
    # show plot
    #pyplot.legend()
    #pyplot.show()

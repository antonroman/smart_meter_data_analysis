#!/usr/bin/python3

# general imports
import sys, getopt
from datetime import datetime


# data processing imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout 

# imports for ploting
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

 

def plot_3d_array(features_set):
    N=8
    features_set = np.random.rand(N, N, N)


    ############################################################################
    # Printing 3D array
    ###########################################################################
    # Create the x, y, and z coordinate arrays.  We use 
    # numpy's broadcasting to do all the hard work for us.
    # We could shorten this even more by using np.meshgrid.
    x = np.arange(features_set.shape[0])[:, None, None]
    y = np.arange(features_set.shape[1])[None, :, None]
    z = np.arange(features_set.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    #c = np.tile(features_set.ravel()[:, None], [1, 2])
    # Do the plotting in a single call.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x.ravel(),
           y.ravel(),
           z.ravel())
    plt.show()

def normalize_numeric_column(field):
    # TODO check if the field exists
    series_power = sample_df[field].astype(np.float64)
    values = series_power.values
    values = values.reshape(len(values),1)
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler = scaler.fit(values)
    values_normalized = scaler.transform(values)
    return values_normalized


def main(argv):
    training_samples_file = ''
    testing_samples_file = ''
    try:
        opts, args = getopt.getopt(argv,"hf:n:",["sample-file=","number-training-samples="])
    except getopt.GetoptError:
        print('lstm_meter_data_consumption_predictor.py -f <sample file> -n <numebr-training samples>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('lstm_meter_data_consumption_predictor.py -f <sample file> -n <number-training samples>')
            sys.exit()
        elif opt in ("-f", "--sample-file"):
            training_samples_file = arg
        elif opt in ("-n", "--number-training-samples"):
            testing_samples_file = arg

    return training_samples_file, testing_samples_file  

if __name__ == "__main__":

    sample_file_path, number_training_samples = main(sys.argv[1:])

    sample_df= pd.read_csv(sample_file_path)

    # print(sample_file['Fh'])
    print(sample_df.head())


    # # iterating the columns 
    # for col in sample_file.columns: 
    #     print(col) 


    # sys.exit(0)

    # print(training_complete)
    # # https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
    # training_complete.dropna(how='all')
    # print(training_complete)

    # ###########################################################################
    # # Data preparation part
    # ###########################################################################

    # Order data by date

    sample_df['Fh'] = pd.to_datetime(sample_df['Fh'])
    sample_df= sample_df.sort_values(by=['Fh'])

    # Beforen standardization we need to check first if the data follows a Gaussian distribution
    # https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    # Two options:
    #   - graphical test
    #   - statistical test

    counts, bins = np.histogram(sample_df['AI'])
    # counts, bins = np.histogram(sample_df['R1'])
    # counts, bins = np.histogram(sample_df['R2'])
    print(bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()

    # data does not follows a Gaussian distribution but Exponential or Pareto

    # Data normalization

    sample_df['AI'] = normalize_numeric_column('AI')
    sample_df['R1'] = normalize_numeric_column('R1')
    sample_df['R4'] = normalize_numeric_column('R4')


    # We are going to predict first power consumption
    # we are going to get 


 # We receive a single file of data so we should split it into training and sample file, 
    # to check teh performance of the model.
    # then with this data I should be able to predict the consumption one week ahead



    # sample_len = len(training_processed)
    # print('Training sample length: ', len(training_processed))
    # features_set = []
    # labels = []
    # N_time_setps = 60

    # # we have to predict a value at time T, based on the data from days T-N where N can be any number of steps
    # # 60 seems to be an optimal value for optimization, so we need to predict the value at day 61st
    # # we need to do sets of 60 values and the lable will be the value at day 61st

    # for i in range(N_time_setps, sample_len): 
    #     print("Iteration number: " + str(i)) 
    #     features_set.append(training_scaled[i-N_time_setps:i,0])
    #     # print("Features set")
    #     # print(features_set)
    #     labels.append(training_scaled[i, 0])
    #     # print("Label")
    #     # print(labels)

    #     # What does this for do?
    #     #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #     # sample_len = 10
    #     # N = 5
    #     # feature_set=[[0,1,2,3,4]] label = [6]
    #     # feature_set=[[0,1,2,3,4],[1,2,3,4,5]] label= [6,7]
    #     # feature_set=[[0,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6]] label = [6,7,8]
    #     # etc

    # # convert lists into numpy arrays
    # features_set, labels = np.array(features_set), np.array(labels)  

    # print("Features set after np")
    # print(features_set)
    # print("Dimensions: " + str(features_set.shape[0]) + "x" + str(features_set.shape[1]))
    # print("Dimensions: " + str(features_set.shape))

   
    # # ANTON
    # # I think the model for daily predictions is not valid for circadian variables, since it has a 
    # # cycle of 24 hours

  

    # # print("training scaled")
    # # print(training_scaled)



    # ###########################################################################
    # # LSTM model setup and training part
    # ###########################################################################
    # # Input must be three-dimensional, comprised of samples, time steps, and features in that order.
    # # data must to be shaped to be accepted by LSTM
    # # This is what the LSTN book say: 

    # # 1st dimension number of records in the dataset, 2n dimension number of time steps
    # features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1)) 
    # # print("Features set after reshape")
    # print("feature_Set dimension after reshape" + str(features_set.shape))

    #  # instance sequential class
    # model = Sequential()  

    # # Creating LSTM and Dropout Layers by adding them to the model
    # # units: 
    # # return_sequences: 
    # # input_shape: argument that expects a tuple containing the number of time steps and the number of features (in this case only one, the power consumption)
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))

    # # LSTMs can quickly converge and even overfit on some sequence prediction problems. To counter
    # # this, regularization methods can be used. Dropout randomly skips neurons during training,
    # # forcing others in the layer to pick up the slack. It is simple and e↵ective. Start with dropout.
    # # Dropout rates between 0.0 (no dropout) and 1.0 (complete dropout) can be set on LSTM layers
    # # with the two di↵erent arguments:
    # # - dropout: dropout applied on input connections.
    # # - recurrent dropout: dropout applied to recurrent connections.
    # model.add(Dropout(0.2))  

    # # add more layers (3 of them, 4 in total)
    # # ANTON test what happens with less layers
    # model.add(LSTM(units=50, return_sequences=True))  
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50, return_sequences=True))  
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50))  
    # model.add(Dropout(0.2))  

    # # A fully connected layer that often follows LSTM layers and is used for outputting a prediction is called Dense().
    # model.add(Dense(units = 1))  

    # # model compilation, using mean squared error for optimization
    # print("Compiling model...")
    # model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # # algortihm training
    # print("Training model...")
    # # epoch: number of iterations first test with 10 and then try with 100
    # # batch_size: 
    # model.fit(features_set, labels, epochs = 20, batch_size = 32)

    # ###########################################################################
    # # Reading data and applying trained LSTM model
    # ###########################################################################

    # # reading data from file and get only the power value columns
    # testing_complete = pd.read_csv(testing_samples_file) 
    # testing_processed = testing_complete.iloc[:, 1:2].values 

    # print(testing_complete)
    # print(testing_processed)
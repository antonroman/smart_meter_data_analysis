#!/usr/bin/python3

# general imports
import sys, getopt, re
from datetime import datetime


# data processing imports
import pandas as pd
from pandas.api.types import CategoricalDtype
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
import seaborn as sns 
sns.set(style="darkgrid")



# ./lstm_meter_data_consumption_forecast/lstm_meter_data_consumption_plotter.py -s data/datoscontadores_csv/S02/meter_data_ZIV0035301588.csv -d data/datoscontadores_csv/S05/meter_data_ZIV0035301588_S05.csv -e data/datoscontadores_csv/Errors/meter_data_ZIV0035301588_error.csv -o data/datoscontadores_plots/
# cd data/datoscontadores_csv/S02
# for f in *.csv; do ../../../lstm_meter_data_consumption_forecast/lstm_meter_data_consumption_plotter.py -s "$f" -d ../S05/"${f%.csv}_S05.csv" -e ../Errors/"${f%.csv}_error.csv" -o ../../datoscontadores_plots/ ;done


def normalize_numeric_column(field):
    # TODO check if the field exists
    series_power = samples_S02_df[field].astype(np.float64)
    values = series_power.values
    values = values.reshape(len(values),1)
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler = scaler.fit(values)
    values_normalized = scaler.transform(values)
    return values_normalized


def main(argv):
    training_samples_file_S02 = ''
    training_samples_file_S05 = ''
    error_file = ''

    testing_samples_file = ''
    try:
        opts, args = getopt.getopt(argv,"hs:d:e:o:",["sample-file-S02=","sample-file-S05=","error-file","output-folder"])
    except getopt.GetoptError:
        print('lstm_meter_data_consumption_predictor.py -s <sample file S02> -s <sample file S05> -e <error file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('lstm_meter_data_consumption_predictor.py -s <sample file S02> -s <sample file S05> -e <error file>')
            sys.exit()
        elif opt in ("-s", "--sample-file-S02"):
            training_samples_file_S02 = arg
        elif opt in ("-d", "--sample-file-S05"):
            training_samples_file_S05 = arg
        elif opt in ("-e", "--error-file"):
            error_file = arg
        elif opt in ("-o", "--output-folder"):
            output_folder = arg    


    return training_samples_file_S02, training_samples_file_S05, error_file, output_folder  

if __name__ == "__main__":
    sample_S02_file_path, sample_S05_file_path , error_file, output_folder = main(sys.argv[1:])

    samples_S02_df= pd.read_csv(sample_S02_file_path)
    samples_S05_df= pd.read_csv(sample_S05_file_path)
    error_df = pd.read_csv(error_file)

    meter_id = re.compile('.*meter_data_(.*)\.csv').match(sample_S02_file_path).group(1)
    print('Processing data for meter ID: ' + meter_id)

    # print(sample_file['Fh'])
    print(samples_S02_df.head())
    print(samples_S05_df.head())


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

    samples_S02_df['Fh'] = pd.to_datetime(samples_S02_df['Fh'])
    samples_S02_df= samples_S02_df.sort_values(by=['Fh'])

    samples_S05_df['Fh'] = pd.to_datetime(samples_S05_df['Fh'])
    samples_S05_df= samples_S05_df.sort_values(by=['Fh'])

    error_df['t'] = pd.to_datetime(error_df['t'])
    error_df= error_df.sort_values(by=['t'])


    # Beforen standardization we need to check first if the data follows a Gaussian distribution
    # https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    # Two options:
    #   - graphical test
    #   - statistical test

    # check data types
    print(samples_S02_df.dtypes)
    print(samples_S05_df.dtypes)
    print(error_df.dtypes)

    # counts, bins = np.histogram(samples02_df['AI'])
    # # counts, bins = np.histogram(samples02_df['R1'])
    # # counts, bins = np.histogram(samples02_df['R2'])
    # print(bins)
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.show()

    # data does not follows a Gaussian distribution but Exponential or Pareto
    # ANTON: check the histogram for the same time for every day, I think this should be a Gaussian distribution

    # set date as index
    samples_S02_df = samples_S02_df.set_index('Fh')
    samples_S05_df = samples_S05_df.set_index('Fh')
    error_df = error_df.set_index('t')

    #samples02_df.timedelta_range(0, periods=9, freq="W")


    # error_df = error_df.set_index('t')
    # ErrCode_type = CategoricalDtype(categories=[1,2], ordered=False)
    # ErrCat_type = CategoricalDtype(categories=[1,2], ordered=False)
    # type_type = CategoricalDtype(categories=['S02','S05'], ordered=False)
    # error_df['ErrCode'] = error_df['ErrCode'].astype(ErrCode_type)
    # error_df['ErrCat'] = error_df['ErrCat'].astype(ErrCat_type)
    # error_df['type'] = error_df['type'].astype(type_type)

    # #error_data_plot = sns.load_dataset(error_df)
    # #sns.catplot(x=error_df.type,y=error_df.ErrCode)
    # #sns.relplot(x=error_df.t,y=error_df.ErrCode,data=error_df)
    # sns.countplot(error_df['type']).set_title("Experinment")

    # ErrCat is alwasy fixed to 2 so we are not going to consider it as variable
    # time serie showing 'ErrCode' and the 'type' will be represented with color
    #  https://seaborn.pydata.org/tutorial/relational.html#relational-tutorial
    #  https://seaborn.pydata.org/tutorial/categorical.html

    # Error Histogram
    # 

    print(error_df.dtypes)


    # sns.set(rc={'figure.figsize':(11, 4)})
    # samples02_df['AI'].plot(linewidth=0.3)
    # samples02_df['R1'].plot(linewidth=0.4)
    # samples02_df['R4'].plot(linewidth=0.4)
    # samples02_df['AI'].set_ylabel('Hourly Consumption (Wh)')
    # samples02_df['R1'].set_ylabel('Hourly reactive power quadrant I (VArh)')
    # samples02_df['R4'].set_ylabel('Hourly reactive power quadrant IV (VArh)')


    cols_plot = ['AI', 'R1', 'R4']
    #axes = samples02_df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
    axes_S02 = samples_S02_df[cols_plot].plot(linewidth=0.3,alpha=0.5,figsize=(11, 9), subplots=True, title="S02 Meter ID: "+meter_id)
    axes_S02[0].set_ylabel('Hourly consumption (Wh)')
    axes_S02[1].set_ylabel('Hourly reactive QI (VArh)')
    axes_S02[2].set_ylabel('Hourly reactive QIV (VArh)')
    plt.savefig(output_folder + meter_id + 'S02.png')

    cols_S05_plot_total = ['AI-Total', 'R1-Total', 'R4-Total']#,'AI-1','R1-1','R4-1','AI-2','R1-2','R4-2']
    axes_S05_Total = samples_S05_df[cols_S05_plot_total].plot(marker='.', alpha=0.1, linestyle='None',figsize=(11, 15), subplots=True, title="S05 Total Meter ID: "+meter_id)
    axes_S05_Total[0].set_ylabel('Daily consumption (KWh)')
    axes_S05_Total[1].set_ylabel('Daily reactive QI (VArh)')
    axes_S05_Total[2].set_ylabel('Daily reactive QIV (VArh)')
    plt.savefig(output_folder + meter_id + 'S05_0_total.png')

    axes_S05_Total = samples_S05_df.diff(periods=1,axis=0)[cols_S05_plot_total].plot(linewidth=0.4, alpha=0.5,figsize=(11, 15), subplots=True, title="S05 consumption per day Meter ID: "+meter_id)
    axes_S05_Total[0].set_ylabel('Daily consumption (KWh)')
    axes_S05_Total[1].set_ylabel('Daily reactive QI (VArh)')
    axes_S05_Total[2].set_ylabel('Daily reactive QIV (VArh)')
    plt.savefig(output_folder + meter_id + 'S05_0_day.png')

    print("S05 Delta Dataframe")
    print(samples_S05_df.diff(periods=1,axis=0).head())

    # GARDAR ISTO COMO CSV PARA REVISAR OS DATOS!!!!!!



    cols_S05_plot_1 = ['AI-1','R1-1','R4-1']
    axes_S05_1 = samples_S05_df[cols_S05_plot_1].plot(marker='.', alpha=0.1, linestyle='None',figsize=(11, 15), subplots=True, title="S05 Tarifa 1 Meter ID: "+meter_id)
    axes_S05_1[0].set_ylabel('Daily consumption (KWh)')
    axes_S05_1[1].set_ylabel('Daily reactive QI (VArh)')
    axes_S05_1[2].set_ylabel('Daily reactive QIV (VArh)')
    plt.savefig(output_folder + meter_id + 'S05_1.png')


    cols_S05_plot_2 = ['AI-2','R1-2','R4-2']
    axes_S05_2 = samples_S05_df[cols_S05_plot_2].plot(marker='.', alpha=0.1, linestyle='None',figsize=(11, 15), subplots=True, title="S05 Tarifa 2 Meter ID: "+meter_id)
    axes_S05_2[0].set_ylabel('Daily night consumption (KWh)')
    axes_S05_2[1].set_ylabel('Daily night reactive QI (VArh)')
    axes_S05_2[2].set_ylabel('Daily night reactive QIV (VArh)')
    plt.savefig(output_folder + meter_id + 'S05_2.png')


    #cols_plot_error = ['ErrCode', 'type']
    #axes_error = error_df[cols_plot_error].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), title="Errors meter ID: "+meter_id)


    # for ax in axes:
    #     ax.set_ylabel('Hourly Totals (Wh)')
    #plt.show()
    #plt.savefig(meter_id+'.png')


    #ANTONNNNNNNNNN  PLOT S05



    # Data normalization

    samples_S02_df['AI'] = normalize_numeric_column('AI')
    samples_S02_df['R1'] = normalize_numeric_column('R1')
    samples_S02_df['R4'] = normalize_numeric_column('R4')


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
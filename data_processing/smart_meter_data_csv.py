#!/usr/bin/python3
# TODO we are considering the output file has .csv extension and we should not
# Data in the CSVs is out of order so we need to order it first
# To process all the files in a folder:
#    for f in data/datoscontadores/*json; do  ./smart_meter_data_csv.py -i "$f" -o "${f%.json}.csv"; done

import json
import csv
import getopt
import sys
import re

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('smart_meter_data_csv.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('smart_meter_data_csv.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    return inputfile, outputfile

if __name__ == "__main__":
    inputfile=''
    outputfile=''
    inputfile, outputfile = main(sys.argv[1:])

    with open(inputfile) as json_file:
        data = json.load(json_file)

        s02Globalreport = data['timeline']['S02']
        s05Globalreport = data['timeline']['S05']
        errorReport = data['timeline']['error']

    ##################################################
    # S02 hourly reports into CSV file
    ##################################################
    with open(outputfile, mode='w') as hourly_consumption_file:
        hourly_consumption_writer = csv.writer(hourly_consumption_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hourly_consumption_file.write('Fh,AI,R1,R4,Bc\n')
        for S02HourlyReport in s02Globalreport:
              
            #'AE' (exported energy) is always 0 in the original samples it is not included
            #'R2' (reactive energy quadrant II) is always 0 in the original samples so it is not included
            #'R3' (reactive energy quadrant III) is always 0 in the original samples so it is not included
            # Do not include data with Byet Control greater than 0x80 (Invalid data)
            if (hex(S02HourlyReport['Bc'])<hex(80)):
                hourly_consumption_writer.writerow([S02HourlyReport['Fh'],S02HourlyReport['AI'],S02HourlyReport['R1'],S02HourlyReport['R4'],S02HourlyReport['Bc']])
            else:
                print(S02HourlyReport['Bc'])

    ##################################################
    # S05 daily reports into CSV file
    ##################################################
    daily_file = re.sub(r'\.csv', r'_S05.csv', outputfile)
    print(daily_file)
    with open(daily_file, mode='w') as daily_consumption_file:
        daily_consumption_writer = csv.writer(daily_consumption_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # For the daily total ("Pt" :0) and for day "Pt" :1 and night "Pt": 2
            # "AIa": Total energy imported
            # "R1a": Reactive energy quadrant I
            # "R4a": Reactive energy quadrant IV
        daily_consumption_file.write('Fh,AI-Total,R1-Total,R4-Total,AI-1,R1-1,R4-1,AI-2,R1-2,R4-2\n')
        Pt_counter = 0
        # dicctionary to store all the values for different Pt at the same day
        S05_sameDay_dict = {}
        # List of billing periods (periodos de tarificacion) whose values we need to get (the rest of them are expected to be 0)
        # "Pt": 0 is the day total
        Pt_list = [1, 2]

        for S05_entry in s05Globalreport:
            Pt = S05_entry['Pt']
            #print('Iteration!')
            if Pt == 0 :
                S05_sameDay_dict['Fh'] = S05_entry['Fh']
                S05_sameDay_dict['AI-Total'] = S05_entry['Value']['AIa']
                S05_sameDay_dict['R1-Total'] = S05_entry['Value']['R1a']
                S05_sameDay_dict['R4-Total'] = S05_entry['Value']['R4a']
            elif Pt in Pt_list  :
                S05_sameDay_dict['AI-'+str(Pt)] = S05_entry['Value']['AIa']
                S05_sameDay_dict['R1-'+str(Pt)] = S05_entry['Value']['R1a']
                S05_sameDay_dict['R4-'+str(Pt)] = S05_entry['Value']['R4a']
                if Pt == Pt_list[-1]:
                    daily_consumption_writer.writerow([S05_sameDay_dict['Fh'],S05_sameDay_dict['AI-Total'],S05_sameDay_dict['R1-Total'],\
                        S05_sameDay_dict['R4-Total'],S05_sameDay_dict['AI-1'],S05_sameDay_dict['R1-1'],S05_sameDay_dict['R4-1'],\
                           S05_sameDay_dict['AI-2'],S05_sameDay_dict['R1-2'],S05_sameDay_dict['R4-2'] ])
    
    ##################################################
    # error reports into CSV files
    ##################################################
    
    # Create error csv file to store all the errors in the report
    output_error_file = re.sub(r'\.csv', r'_error.csv', outputfile)
    with open(output_error_file, mode='w') as error_file:
        error_writer = csv.writer(error_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        error_file.write('t,ErrCat,ErrCode,type\n')
        for error_entry in errorReport:
            error_writer.writerow([error_entry['t'],error_entry['ErrCat'],error_entry['ErrCode'],error_entry['type']])
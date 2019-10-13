#!/usr/bin/python3
import json
import csv
import getopt
import sys

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

    s02list=[]

    with open(inputfile) as json_file:
        data = json.load(json_file)

        s02Globalreport = data['timeline']['S02']


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
    
    # Data in the CSV is out of order
    # To process all the files in a folder:
    #       for f in data/datoscontadores/*json; do  ./smart_meter_data_csv.py -i "$f" -o "${f%.json}.csv"; done

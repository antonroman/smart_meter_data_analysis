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
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    return inputfile, outputfile

if __name__ == "__main__":
    #./smart_meter_parsing.py data/meter_data.json hourly_consumption.csv
    inputfile=''
    outputfile=''
    inputfile, outputfile = main(sys.argv[1:])

    s02list=[]

    with open(inputfile) as json_file:
        data = json.load(json_file)
        # store day consumption in hourly_consumption
        i=0
        for day_report in data:
            s02list.append(data[i]['timeline']['S02'])
            #print(json.dumps(data[i]['timeline']['S02']))
            i=i+1

    with open(outputfile, mode='w') as hourly_consumption_file:
        hourly_consumption_writer = csv.writer(hourly_consumption_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for s02Daylyreport in s02list:
            #hourly_consumption_writer.writerow({'Kona'})
            for S02HourlyReport in s02Daylyreport:
                # print(S02HourlyReport)
                # print(S02HourlyReport['Fh'])
                # print(S02HourlyReport['AI'])
                print([S02HourlyReport['Fh'],S02HourlyReport['AI']])
                hourly_consumption_writer.writerow([S02HourlyReport['Fh'],S02HourlyReport['AI']])


    #parsed_json = (json.loads(json_data))
    #print(json.dumps(data, indent=4, sort_keys=True))
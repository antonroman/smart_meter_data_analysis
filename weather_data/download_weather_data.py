from math import sqrt
from datetime import datetime as dt
import requests
import pandas as pd

# AEMET Open Data API URL and credentials
aemet_base_url = 'https://opendata.aemet.es/opendata'
aemet_api_key = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJndWlsbGVybW8uYmFycmVpcm9AZGV0LnV2aWdvLmVzIiwianRpIjoiOTYwNjA3NGQtNjBmYy00MWE4LThlMzQtMGNiY2MzODkzYmRiIiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE2MDY4MzY1NjcsInVzZXJJZCI6Ijk2MDYwNzRkLTYwZmMtNDFhOC04ZTM0LTBjYmNjMzg5M2JkYiIsInJvbGUiOiIifQ.JyL4G-tCZZIWRsJp5HBOMNdkPWE1rTS3vTkBu2CdI6c'
request_header = {'api_key': aemet_api_key}

# Get all weather stations in Spain
stations_list_first_request = requests.get(aemet_base_url + '/api/valores/climatologicos/inventarioestaciones/todasestaciones', headers=request_header)
if stations_list_first_request.status_code == 200:
    # Get the URL to get the list of stations
    stations_list_url = stations_list_first_request.json()['datos']
    stations_list_final_request = requests.get(stations_list_url)
    if stations_list_final_request.status_code == 200:
        # Get the list of stations
        stations_list = stations_list_final_request.json()

# Parse the latitude and longitude strings from the stations
def transform_coordinates(station):
    latitude_sign = 1 if station['latitud'][-1] == 'N' else -1
    longitude_sign = 1 if station['longitud'][-1] == 'E' else -1
    latitude = latitude_sign * float(station['latitud'][:-1])/(10**4)
    longitude = longitude_sign * float(station['longitud'][:-1])/(10**4)
    station['latitud'] = latitude
    station['longitud'] = longitude
    return station

stations_list = list(map(transform_coordinates, stations_list))

# Search for the closest station to our meters in Puerto Real (Cádiz)
meters_coordinates = {'lat': 36.533430217658996, 'long': -6.187244367031206}

def calculate_distance(station):
    return sqrt((station['latitud']-meters_coordinates['lat'])**2 + (station['longitud']-meters_coordinates['long'])**2)

stations_distance = ([{'id': x['indicativo'], 'distance': calculate_distance(x)} for x in stations_list])
stations_distance.sort(key=lambda x: x['distance'])
closest_station = stations_distance[0]['id']

# Get the weather for the closest station
aemet_date_format = '%Y-%m-%dT%H:%M:%SUTC'
from_date = dt(2019, 5, 1) # the smart meters data begins in May or June 2019
to_date = dt(2021, 4, 1) # the smart meters data last to March 2019
historical_weather_first_request = requests.get(aemet_base_url + f'/api/valores/climatologicos/diarios/datos/fechaini/{from_date.strftime(aemet_date_format)}/fechafin/{to_date.strftime(aemet_date_format)}/estacion/{closest_station}', headers=request_header)
if historical_weather_first_request.status_code == 200:
    # Get the URL to get the list of stations
    historical_weather_url = historical_weather_first_request.json()['datos']
    historical_weather_last_request = requests.get(historical_weather_url)
    if historical_weather_last_request.status_code == 200:
        # Get the weather data
        weather_data = historical_weather_last_request.json()

# Change the Spanish decimal separator (',') to the American one ('.')
for item in weather_data:
    for key in item:
        item[key] = item[key].replace(',', '.')

# Filter the data and write it to a CSV file
csv_output_name = 'weather.csv'
weather_df = pd.DataFrame(weather_data)
weather_df = weather_df[['fecha', 'tmed', 'tmin', 'tmax']].rename(columns={'fecha': 'date', 'tmed': 'average_temperature', 'tmin': 'lowest_temperature', 'tmax': 'highest_temperature'})
weather_df[['average_temperature', 'lowest_temperature', 'highest_temperature']] = weather_df[['average_temperature', 'lowest_temperature', 'highest_temperature']].astype("float")
weather_df.to_csv(csv_output_name, index=False)

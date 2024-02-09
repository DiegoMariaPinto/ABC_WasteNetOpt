# File di funzioni che utilizzano Open Street Map Routing Machine (OSRM) per il calcolo delle distanze e dei tempi

import pandas as pd
import requests as rq
import numpy as np
import json
from pathlib import Path
from ast import literal_eval
##########################################################################################
##########################################################################################
#########################  DOCKER SERVER on PORT 5000  ###################################
##########################################################################################
##########################################################################################

# before to go please start the map service on docker with the following command
# docker run -t -i -p 5000:5000 -v $(pwd):/data osrm/osrm-backend osrm-routed --algorithm mld /data/italy-latest.osrm
# tipo di servizio: 'OPENROUTESERVICE' si collega a https://openrouteservice.org/
# 'LOCAL' si collega al server locale installato tramite osrm-backend project
route_service = 'LOCAL'
# Inserire qui la propria api key ottenuta da https://openrouteservice.org/
# serve solo se il tipo di servizio è 'OPENROUTESERVICE
myApyKey = '5b3ce3597851110001cf62488cc89bf95d4b4c019b17589365548095'
DEBUG = False

def getPath(long1, lat1, long2, lat2):
    if route_service == 'OPENROUTESERVICE':
        headers = {
            'Accept': 'application/json; charset=utf-8'
        }
        myurl = 'https://api.openrouteservice.org/directions?api_key=' + myApyKey + '&coordinates=' + str(
            long1) + ',' + str(lat1) + '%7C' + str(long2) + ',' + str(lat2) + '&profile=driving-car'
        response = rq.request(url=myurl, headers=headers, method='GET')
        return response
    elif route_service == 'LOCAL':
        myurl = 'http://127.0.0.1:5000/route/v1/driving/'
        myurl += str(long1) + ',' + str(lat1) + ';' + str(long2) + ',' + str(lat2)
        params = (
            ('steps', 'false'),
        )
        response = rq.get(myurl, params=params)
        return response
    else:
        return 'ERROR'


def getDistance(response):
    a = dict(response.json())
    if DEBUG:
        print(a)
    if route_service == 'OPENROUTESERVICE':
        return a['routes'][0]['summary']['duration']
    elif route_service == 'LOCAL':
        dist = a['routes'][0]['distance']
        return dist
    else:
        return 'ERROR'


def getTime(response):
    a = dict(response.json())
    if DEBUG:
        print(a)
    if route_service == 'OPENROUTESERVICE':
        return a['routes'][0]['summary']['duration']
    elif route_service == 'LOCAL':
        duration = a['routes'][0]['duration']
        return duration
    else:
        return 'ERROR'


##############################################################################################

def OSM(df):

    print(df)
    n = len(df.index)

    print("number of rows: ", n)

    A = df.values  # matrice n * 3

    df_dis = df.copy()
    df_dur = df.copy()

    dis_dict = {}
    dur_dict = {}

    for i in range(n):
        new_col_dis = []
        new_col_dur = []
        id_from = A[i][0]
        for j in range(n):
            id_to = A[j][0]

            long1 = A[i][2]
            lat1 = A[i][1]
            long2 = A[j][2]
            lat2 = A[j][1]

            if i != j:
                print("distance from {},{} to {},{}".format(long1, lat1, long2, lat2))
                response = getPath(long1, lat1, long2, lat2)
                dis = getDistance(response) / 1000
                dur = (getTime(response) / 3600) / (0.7)
            else:
                dis = 0
                dur = 0
            print("distance from {} to {}: {}".format(id_from, id_to, dis))
            new_col_dis.append(dis)
            print("duration from {} to {}: {}".format(id_from, id_to, dur))
            new_col_dur.append(dur)

            dis_dict[(id_from,id_to)] = dis
            dur_dict[(id_from,id_to)] = dur

        df_dis[id_from] = new_col_dis
        df_dur[id_from] = new_col_dur

    # La matrice viene creata con colonna origine - riga destinazione
    # Quindi viene trasposta per leggerla come riga orgine - colonna destinazione
    distance = df_dis.iloc[:, 3:].T
    duration = df_dur.iloc[:, 3:].T

    return distance, duration, dis_dict, dur_dict




if __name__ == "__main__":

    path = '/home/administrator/PycharmProjects/OptNetDesign/Data/'
    #path = str(Path().resolve().parent) + '\\Data\\'

    df_service = pd.read_excel(path + 'real_data.xlsx', sheet_name='Service',  converters={'service_id': str, 'lat': float, 'long': float})[['service_id', 'lat', 'long']]
    df_depot    = pd.read_excel(path+'real_data.xlsx', sheet_name='Depot',      converters={'depot_id': str, 'lat': float, 'long': float})[['depot_id', 'lat', 'long']]
    df_disposal = pd.read_excel(path+'real_data.xlsx', sheet_name='Disposal' , converters={'disposal_id'    :str,'lat':float,'long':float})[['disposal_id','lat','long']]
    df_tranship = pd.read_excel(path+'real_data.xlsx', sheet_name='Transship', converters={'transhipment_id':str,'lat':float,'long':float})[['transhipment_id','lat','long']]

    df = pd.DataFrame(np.concatenate((df_service.values, df_depot.values, df_disposal.values, df_tranship.values), axis=0))
    df.columns = ['node_id', 'lat', 'long']

    df = df.astype({"node_id": str,'lat':float,'long':float}, errors='raise')

    distance, duration, dis_dict, dur_dict = OSM(df)

    # Save distance and duration dictionaries:
    # 1)
    # convert each tuple key to a string before saving as json object
    dis_dict_to_save = {str(k): v for k, v in dis_dict.items()}
    dur_dict_to_save = {str(k): v for k, v in dur_dict.items()}

    # 2) dump dictionary as a .json
    with open(path+'dis_dict.json', 'w') as f:
        json.dump(dis_dict_to_save, f)
    with open(path+'dur_dict.json', 'w') as f:
        json.dump(dur_dict_to_save, f)

    print('dictionaries have been saved to local memory')

    # load dis/dur dictionaries in two stages:
    # 1) load json object
    with open(path+'dis_dict.json', 'r') as json_file:
        dis_data = json.load(json_file)
    with open(path+'dur_dict.json', 'r') as json_file:
        dur_data = json.load(json_file)
    # 2) convert loaded keys from string back to tuple
    dis_dict = {literal_eval(k): v for k, v in dis_data.items()}
    dur_dict = {literal_eval(k): v for k, v in dur_data.items()}

    print('dis. & dur. dictionaries have been loaded')






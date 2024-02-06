import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import requests
from entsoe import EntsoePandasClient

def load_data():

    df = pd.read_csv("datasets/energy_dataset.csv")

    #drop columns
    columns = ["generation hydro pumped storage aggregated", "forecast wind offshore eday ahead", "generation fossil coal-derived gas", "generation wind offshore", "generation marine", "generation geothermal",
    "generation fossil peat","generation fossil oil shale","forecast solar day ahead","forecast wind onshore day ahead","total load forecast","price day ahead"]
    df = df.drop(columns,axis = 1)

    #fill empty entries
    df.fillna(df.interpolate(method="linear"),inplace=True)

    #format time and set as index
    df["time"] = pd.to_datetime(df["time"],format = "ISO8601")
    df['time'] = df['time'].apply(lambda x: x.replace(tzinfo=None))
    df["time"] = pd.to_datetime(df["time"],format="ISO8601")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["week"] = df["time"].dt.weekday
    df = df.set_index('time')

    return df

def split_data(dataframe):
    df = dataframe
    x = df.drop("total load actual",axis = 1)
    y = df["total load actual"]
    tss = TimeSeriesSplit(n_splits=2)
    print(tss)
    for train,test in tss.split(df):
        trainx, testx = x.iloc[train,:], x.iloc[test,:]
        trainy, testy = y.iloc[train], y.iloc[test]

    return trainx,testx,trainy,testy


def energy_api(starttime,endtime):
    #response = requests.get("https://apidatos.ree.es/es/datos/demanda/demanda-maxima-horaria?start_date=2024-01-01T00:00&end_date=2024-01-31T23:59&time_trunc=hour")
    client = EntsoePandasClient(api_key= "b337a1d6-b64c-49db-ac5a-8a260d29ec52")
    #year=2017, month=1, day=1, hour=0
    start = pd.Timestamp(starttime, tz = 'Europe/Madrid' )
    end = pd.Timestamp(endtime, tz = 'Europe/Madrid' )
 
    energy = client.query_generation("ES", start = start, end = end)
    load  = client.query_load("ES", start = start, end=end)
    return pd.concat([energy,load],axis=1)
  
    


#df = load_data()    
#print(df.columns)
"""
df = load_data()

xt,xtest,yt,ytest = split_data(df)
print(xt.columns)
print(yt[0])"""
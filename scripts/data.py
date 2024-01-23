import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


def load_data():

    df = pd.read_csv("datasets/energy_dataset.csv")

    #drop columns
    columns = ["generation hydro pumped storage aggregated", "forecast wind offshore eday ahead", "generation fossil coal-derived gas", "generation wind offshore", "generation marine", "generation geothermal",
    "generation fossil peat","generation fossil oil shale"]
    df = df.drop(columns,axis = 1)

    #fill empty entries
    df.fillna(df.interpolate(method="linear"),inplace=True)

    #format time and set as index
    df["time"] = pd.to_datetime(df["time"],format = "%Y-%m-%d %H:%M:%S")
    df['time'] = df['time'].apply(lambda x: x.replace(tzinfo=None))
    df["time"] = pd.to_datetime(df["time"],format="ISO8601")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df = df.set_index('time')

    return df

def testtrainsplit(dataframe):
    df = dataframe
    x = df.drop("total load actual",axis = 1)
    y = df["total load actual"]
    tss = TimeSeriesSplit(n_splits=2)
    print(tss)
    for train,test in tss.split(df):
        trainx, testx = x.iloc[train,:], x.iloc[test,:]
        trainy, testy = y.iloc[train], y.iloc[test]

    return trainx,testx,trainy,testy

#df = pd.read_csv("datasets/weather_features.csv")
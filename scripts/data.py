import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from entsoe import EntsoePandasClient



def energy_api(starttime,endtime,csv = None):
    """
    Makes queries to the Entsoe API for the given timespan. 
    :param starttime: Query start time
    :param endtime: Query end time
    :param csv: name of csv file to convert and download dataframe into
    :return: Pandas dataframe of load and generation data
    """
    
    client = EntsoePandasClient(api_key= "b337a1d6-b64c-49db-ac5a-8a260d29ec52")
    start = pd.Timestamp(starttime, tz = 'Europe/Madrid' )
    end = pd.Timestamp(endtime, tz = 'Europe/Madrid' )
 
    energy = client.query_generation("ES", start = start, end = end)
    load  = client.query_load("ES", start = start, end=end)
    df = pd.concat([energy,load],axis=1)
    df["time"] = df.index
    df = df[df['time'].dt.minute == 0]
    df.index = df["time"]
    df = df.drop(["time"],axis = 1)
    if csv:
        f = open(csv, "w")
        f.truncate()
        f.close()
        df.to_csv(csv, index=True,index_label="time")
    return df

def load_data(dataset = None,daily = None, start = None, end = None,csv = None):

    if dataset:
        df = pd.read_csv(dataset)
    elif start:
        df = energy_api(start,end, csv)
    else:
        df = pd.read_csv("datasets/energy_dataset.csv")
    return df

def preprocessing(dataframe,columns = None,daily = None):
    
    df = dataframe
    if columns:
        df = dataframe.drop(columns, axis=1, errors='ignore')
    df["time"] = pd.to_datetime(df["time"], utc = True)
    if daily:
        return df.resample("D").mean()
    df = df.set_index('time').asfreq("1H")
    df.fillna(df.interpolate(method="linear"),inplace=True)
    time = df.index
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["week"] = df.index.weekday
    return df

def check_missing_values(dataframe):
    
    new_df = pd.DataFrame(
        pd.date_range(
            start=dataframe.index.min(), 
            end=dataframe.index.max(),
            freq='H'
        ).difference(dataframe.index)
    )
    missing_columns = [col for col in dataframe.columns if col!="time"]
    print(missing_columns)
    
    # add null data
    new_df[missing_columns] = np.nan

    # fix column names
    new_df.columns = ["time"] + missing_columns
    
    return new_df



def split_data(dataframe,target,train,validation):
    df = dataframe
    x = df.drop(target,axis = 1)
    y = df[target]
    end_train =  train
    end_validation = validation
    trainx,trainy = x.loc[: end_train, :],y.loc[: end_train]
    valx,valy   = x.loc[end_train:end_validation, :],y.loc[end_train:end_validation]
    testx,testy  = x.loc[end_validation:, :],y.loc[end_validation:]
    return trainx,testx,trainy,testy, end_validation

"""
#"2017-06-30 23:00:00+00:00"
#'2018-03-31 23:00:00+00:00'
df = load_data()
df = preprocessing(df)
trainx,testx,trainy,testy, end_validation = split_data(df,"total load actual","2017-06-30 23:00:00+00:00",'2018-03-31 23:00:00+00:00')
print("complete")"""

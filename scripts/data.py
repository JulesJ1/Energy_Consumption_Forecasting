import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from entsoe import EntsoePandasClient
import requests

from astral.sun import sun
from astral import LocationInfo
from datetime import datetime,timedelta


def energy_api(starttime,endtime = None,csv = None):
    """
    Makes queries to the Entsoe API for the given timespan. 

    Args:
        starttime: Query start time
        endtime: Query end time
        csv: name of csv file to convert and download dataframe into
    Returns:
        A pandas dataframe of hourly load and generation data
    """
    
    client = EntsoePandasClient(api_key= "b337a1d6-b64c-49db-ac5a-8a260d29ec52")
    start = pd.Timestamp(starttime, tz = 'Europe/Madrid' )

    if endtime:
        end = pd.Timestamp(endtime, tz = 'Europe/Madrid' )
        energy = client.query_generation("ES", start = start, end = end)
        load  = client.query_load("ES", start = start, end=end)
    else:
        energy = client.query_generation("ES", start = start)
        load  = client.query_load("ES", start = start)

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

def weather_api():
    r = requests.get("https://api.openweathermap.org/data/3.0/onecall?lat=40.416775&lon=-3.703790&exclude=current,minutely,daily,alerts&appid=55194385c9175281575e04c124e2fdbc")

    r = r.json()

    df = pd.json_normalize(r["hourly"])
    df = pd.concat([df["dt"],df["temp"]],axis=1)
    df["dt"] = pd.to_datetime(df["dt"],unit='s')
    df = df.set_index('dt').asfreq("1H")
    
    return df


def weather_data(csv = None,steps = None):
    """

    
    
    """
    if csv:
        
        weather_raw = pd.read_csv("../../weatherhistoric.csv")
        f = open("../datasets/weather_updated.csv", "w")
        f.truncate()
        f.close()

        skiplength = 385
        freq = steps
        curr=0

        while(curr < len(weather_raw)):
            if curr == 0:
                weather_raw.iloc[curr:curr + freq,:].to_csv("../datasets/weather_updated.csv", mode="a",header = True)   
            else: 
                weather_raw.iloc[curr:curr + freq,:].to_csv("../datasets/weather_updated.csv", mode="a",header = False)
            curr += skiplength * (steps/6)
            

    weather_df = pd.read_csv("datasets/weather_updated.csv")
    weather_df = weather_df[["slice dt unixtime","temperature"]]
    weather_df = weather_df.rename(columns={"slice dt unixtime":"date","temperature":"temp"})
    weather_df["date"] = pd.to_datetime(weather_df["date"],unit='s')
    weather_df.set_index("date",inplace =True)
    
    return weather_df


def load_data(dataset = None,daily = None, start = None, end = None,csv = None):
    """
    Loads an energy dataset from a csv or the Entsoe API.

    Args:
        dataset: If loading from a csv file then this is the file name.
        daily: Get the daily energy consumption data.
        start: Start datetime of the data request.
        end: End datetime of the data request.
        csv: Name of a csv file if the data should be stored in a file.
    Returns:
        A dataframe containing energy consumption and generation data.
    """

    if dataset:
        df = pd.read_csv(dataset)
    elif start:
        df = energy_api(start,end, csv)
    else:
        df = pd.read_csv("datasets/energy_dataset.csv")
    return df

def preprocessing(dataframe,columns = None,daily = None):
    """
    Applies the needed preprocessing steps to a dataframe. Removes specified columns,
    sets the "time" column as the index. Resamples the data as daily if needed. Creates 
    year, month and week features using the index.

    Args:
        dataframe: The dataframe to be transformed.
        columns: Columns to be removed from the dataframe.
        daily: Resample the data frequency as daily, uses the mean of the hours in each day.
    Returns:
        A pandas dataframe transformed using the preprocessing steps.
    """
    
    df = dataframe
    if columns:
        df = dataframe.drop(columns, axis=1, errors='ignore')
    df["time"] = pd.to_datetime(df["time"], utc = True)
    df["time"] = df["time"].dt.tz_localize(None)
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
    """
    Checks the dataframe for missing rows in  the index

    Args:
        dataframe: A dataframe with missing rows.
    Returns:
        A pandas dataframe with an hour frequency.
    """
    
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
    """
    Splits a time series dataframe into training, validation and test sets.
    Each split is in chronological.

    Args:
        dataframe: A dataframe.
        target: Column that is being predicted.
        train: The timestamp for the end of the training set.
        validation: The timestamp for the end of the validation set.
    Returns:
        trainx: Training set features.
        testx: Test set features.
        trainy: Training set target.
        testy: Test set target
        end_validation: The end of validation set timestamp.
    """
    df = dataframe
    x = df.drop(target,axis = 1)
    y = df[target]
    end_train =  train
    end_validation = validation
    trainx,trainy = x.loc[: end_train, :],y.loc[: end_train]
    valx,valy   = x.loc[end_train:end_validation, :],y.loc[end_train:end_validation]
    testx,testy  = x.loc[end_validation:, :],y.loc[end_validation:]
    return trainx,testx,trainy,testy, end_validation


def fourier_features(feature,cycle_length,order):
    result = pd.DataFrame()

    k = 2 * np.pi * feature/cycle_length
    for i in range(1,order+1):
        result[f"sin_{feature.name}_{i}"] =  np.sin(i*k)
        result[f"cos_{feature.name}_{i}"]    =  np.cos(i*k)
    return result
def tempfeatures(dataframe):
    temp_features = dataframe["temp"].copy()
    temp_features = temp_features.to_frame()
    temp_features['temp_roll_mean_1_day'] = temp_features['temp'].rolling(24, closed='left').mean()
    temp_features['temp_roll_mean_7_day'] = temp_features['temp'].rolling(24*7, closed='left').mean()
    temp_features['temp_roll_max_1_day'] = temp_features['temp'].rolling(24, closed='left').max()
    temp_features['temp_roll_min_1_day'] = temp_features['temp'].rolling(24, closed='left').min()
    temp_features['temp_roll_max_7_day'] = temp_features['temp'].rolling(24*7, closed='left').max()
    temp_features['temp_roll_min_7_day'] = temp_features['temp'].rolling(24*7, closed='left').min()

    return temp_features

def create_features(dataframe,weather_df = None):
    #add weahter_api call for live data


    location = LocationInfo(
    name='Washington DC',
    region='Spain',
    timezone='CET')

    calendar_features = pd.DataFrame(index=dataframe.index)
    calendar_features['month'] = calendar_features.index.month
    calendar_features['week_of_year'] = calendar_features.index.isocalendar().week
    calendar_features['week_day'] = calendar_features.index.day_of_week + 1
    calendar_features['hour_day'] = calendar_features.index.hour + 1
    sunrise_hour = [
        sun(location.observer, date=date, tzinfo=location.timezone)['sunrise'].hour
        for date in dataframe.index
    ]
    sunset_hour = [
        sun(location.observer, date=date, tzinfo=location.timezone)['sunset'].hour
        for date in dataframe.index
    ]
    sun_light_features = pd.DataFrame({
                            'sunrise_hour': sunrise_hour,
                            'sunset_hour': sunset_hour}, 
                            index = dataframe.index
                        )
    sun_light_features['daylight_hours'] = (
        sun_light_features['sunset_hour'] - sun_light_features['sunrise_hour']
    )
    sun_light_features['is_daylight'] = np.where(
                                            (dataframe.index.hour >= sun_light_features['sunrise_hour']) & \
                                            (dataframe.index.hour < sun_light_features['sunset_hour']),
                                            1,
                                            0
                                        )
    exo_features = pd.concat([
                                calendar_features,
                                sun_light_features,
                            
                            ], axis=1)

    month_encoded = fourier_features(exo_features["month"], 12,1)
    week_of_year_encoded = fourier_features(exo_features['week_of_year'], 52,1)
    week_day_encoded = fourier_features(exo_features['week_day'], 7,1)
    hour_day_encoded = fourier_features(exo_features['hour_day'], 24,1)
    cyclical_features = pd.concat([
                            month_encoded,
                            week_of_year_encoded,
                            week_day_encoded,
                            hour_day_encoded,
                        ], axis=1)
    if weather_df:
        temp_df = tempfeatures(weather_df)
        temp_df.index = dataframe.index
        cyclical_features = pd.concat([cyclical_features,temp_df], axis = 1)
    #cyclical_features = cyclical_features.join(temp_features["temp"])
    exo_features = pd.concat([exo_features, cyclical_features], axis=1)
    return exo_features
"""
now = datetime.now()
prevhour = now - timedelta(hours=12)
starttime = prevhour.strftime("%d/%m/%Y %H:00:00")
endtime = now.strftime("%Y-%m-%d %H:00:00")
df =energy_api(starttime,endtime=endtime)

print(df.columns)"""

"""
df = load_data()
df = preprocessing(df)
#temp = tempfeatures(df)
weather_df = pd.read_csv("datasets/weather_features.csv")
weather_df = weather_df.loc[weather_df["city_name"] == "Madrid"]
weather_df = weather_df.drop_duplicates(subset="dt_iso")
temp = tempfeatures(weather_df)
print(temp.head)
print(temp.iloc[70])

temp = temp.dropna()
print(len(temp))
exo = create_features(df)

print(len(exo))
print(exo.iloc[-1])
exo = exo.dropna()
print(len(exo))
"""
"""
df = load_data(dataset="datasets/energy_updated.csv")
df = preprocessing(df)
t = tempfeatures(weather_data())
t = t.loc['2024-03-01 00:00:00':'2024-03-01 05:00:00',:] 
print(t.index)
print(df.index)
exo = create_features(df)
print(exo)"""
"""
df = energy_api('2024-03-06 21:00:00','2024-03-06 22:00:00')
print(df["Actual Load"])
#"2017-06-30 23:00:00+00:00"
#'2018-03-31 23:00:00+00:00'
df = load_data()
df = preprocessing(df)
trainx,testx,trainy,testy, end_validation = split_data(df,"total load actual","2017-06-30 23:00:00+00:00",'2018-03-31 23:00:00+00:00')
print("complete")"""

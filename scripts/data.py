import pandas as pd
import numpy as np

from entsoe import EntsoePandasClient
import requests

from astral.sun import sun
from astral import LocationInfo
from sklearn.preprocessing import PolynomialFeatures



def energy_api(starttime,endtime = None,csv = None):
    """
    Makes queries to the Entsoe API for the given timespan. 

    Args:
        starttime: Query start time.
        endtime: Query end time.
        csv: name of csv file to convert and download dataframe into.
    Returns:
        A pandas dataframe of hourly load and generation data.
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
    """
    Makes a queriyto the OpenWeather API. The API returns hourly 48 hour ahead weather predictions. 

  
    Returns:
        A pandas dataframe of hourly 48 hour ahead temperature predictions.
    """    
    r = requests.get("https://api.openweathermap.org/data/3.0/onecall?lat=40.416775&lon=-3.703790&exclude=current,minutely,daily,alerts&appid=55194385c9175281575e04c124e2fdbc")

    r = r.json()

    df = pd.json_normalize(r["hourly"])
    df = pd.concat([df["dt"],df["temp"]],axis=1)
    df["dt"] = pd.to_datetime(df["dt"],unit='s')
    df = df.set_index('dt').asfreq("1H")
    
    return df


def weather_data(csv = None,steps = None):
    """
    Resamples bulk historical weather data from the OpenWeather API, so that it can be used for training a model. Each forecast from the API contains 385 samples.

    Args:
        csv: The name of the csv file to store the resampled weather data.
        steps: The number of samples taken from each forecast.
        
    Returns:
        A pandas dataframe of hourly historical weather forecasts.
    
    
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
    Splits a time series dataframe into training, validation and test sets, while maintaning the datas sequential order.
    

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
    """
    Creates fourier wave patterns for the provided features to represent seasonal trends. 

    Args:
        feature: Seasonal data such as the day of week, month or hour of day.
        cycle_length: The temporal length of the pattern.
        order: The number of cos, sin pairs to generate.
    Returns:
        A dataframe containing sin and cos wave columns representing seasonal patterns.
    """
    result = pd.DataFrame()

    k = 2 * np.pi * feature/cycle_length
    for i in range(1,order+1):
        result[f"sin_{feature.name}_{i}"] =  np.sin(i*k)
        result[f"cos_{feature.name}_{i}"]    =  np.cos(i*k)
    return result


def tempfeatures(dataframe):

    """    
    Creates Rolling temperature features (minimum, maximum, average) for the daily and weekly temperature observations around each sample. 

    Args:
        dataframe: A dataframe containing a temerature column labeled "temp".

    Returns:
        A dataframe containing rolling temperature aggregation from the original data.
    
    """
    temp_features = dataframe["temp"].copy()
    temp_features = temp_features.to_frame()
    temp_features['temp_roll_mean_1_day'] = temp_features['temp'].rolling(24, closed='left').mean()
    temp_features['temp_roll_mean_7_day'] = temp_features['temp'].rolling(24*7, closed='left').mean()
    temp_features['temp_roll_max_1_day'] = temp_features['temp'].rolling(24, closed='left').max()
    temp_features['temp_roll_min_1_day'] = temp_features['temp'].rolling(24, closed='left').min()
    temp_features['temp_roll_max_7_day'] = temp_features['temp'].rolling(24*7, closed='left').max()
    temp_features['temp_roll_min_7_day'] = temp_features['temp'].rolling(24*7, closed='left').min()

    return temp_features

def create_features(dataframe,polynomial_columns, weather_df = None):
    """
        Creates exogonous features derived from a dataframes timestamp:
        - Calender features
        - sunlight hours
        - season trends and patterns 

    Args:
        dataframe: A dataframe with a DateTime index.
        weather_df: If provided, adds temperature features to the resulting dataframe.

    Returns:
        A dataframe exogenous features that are needed to train a model or create predictions.
    """


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


    transformer_poly = PolynomialFeatures(
                        degree           = 2,
                        interaction_only = True,
                        include_bias     = False,
                        

                    ).set_output(transform="pandas")
    """    'sin_sunrise_hour_1',
        'cos_sunrise_hour_1',
        'sin_sunset_hour_1',
        'cos_sunset_hour_1',"""
    poly_cols = [
        'sin_month_1', 
        'cos_month_1',
        'sin_week_of_year_1',
        'cos_week_of_year_1',
        'sin_week_day_1',
        'cos_week_day_1',
        'sin_hour_day_1',
        'cos_hour_day_1',
        'daylight_hours',
        'is_daylight',

    ]
    poly_cols = polynomial_columns

    poly_features = transformer_poly.fit_transform(exo_features[poly_cols].dropna())
    #poly_features = transformer_poly.fit_transform(exog[poly_cols])
    poly_features = poly_features.drop(columns=poly_cols)
    poly_features.columns = [f"poly_{col}" for col in poly_features.columns]
    poly_features.columns = poly_features.columns.str.replace(" ", "__")

    exo_features = pd.concat([exo_features, poly_features], axis=1)



    return exo_features


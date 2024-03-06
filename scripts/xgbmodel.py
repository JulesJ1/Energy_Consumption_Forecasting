from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import data 
from skopt.space import Real, Integer
from skopt import BayesSearchCV
import pandas as pd
from joblib import dump, load
import numpy as np

from skforecast.ForecasterBaseline import ForecasterEquivalentDate
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import check_y

from astral.sun import sun
from astral import LocationInfo


class xgboost_model():
    def __init__(self):
        self.df = data.load_data()
        self.trainx,self.testx,self.trainy,self.testy, self.end_validation = data.split_data

    def split_data(self):
        self.trainx, self.testx, self.trainy, self.testy = data.split_data(self.df)

    def build_pipeline(self):
        estimators = [
            ("reg", XGBRegressor())
        ]

        return Pipeline(steps = estimators)

    def fourier_features(feature,cycle_length,order):
        result = pd.DataFrame()

        k = 2 * np.pi * feature/cycle_length
        for i in range(1,order+1):
            result[f"sin_{feature.name}_{i}"] =  np.sin(i*k)
            result[f"cos_{feature.name}_{i}"]    =  np.cos(i*k)
        return result


    def exo_features(self):
        location = LocationInfo(
        name='Washington DC',
        region='Spain',
        timezone='CET')

        calendar_features = pd.DataFrame(index=df2.index)
        calendar_features['month'] = calendar_features.index.month
        calendar_features['week_of_year'] = calendar_features.index.isocalendar().week
        calendar_features['week_day'] = calendar_features.index.day_of_week + 1
        calendar_features['hour_day'] = calendar_features.index.hour + 1
        sunrise_hour = [
            sun(location.observer, date=date, tzinfo=location.timezone)['sunrise'].hour
            for date in df2.index
        ]
        sunset_hour = [
            sun(location.observer, date=date, tzinfo=location.timezone)['sunset'].hour
            for date in df2.index
        ]
        sun_light_features = pd.DataFrame({
                                'sunrise_hour': sunrise_hour,
                                'sunset_hour': sunset_hour}, 
                                index = df2.index
                            )
        sun_light_features['daylight_hours'] = (
            sun_light_features['sunset_hour'] - sun_light_features['sunrise_hour']
        )
        sun_light_features['is_daylight'] = np.where(
                                                (df2.index.hour >= sun_light_features['sunrise_hour']) & \
                                                (df2.index.hour < sun_light_features['sunset_hour']),
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
        cyclical_features = pd.concat([cyclical_features,temp_features], axis = 1)
        #cyclical_features = cyclical_features.join(temp_features["temp"])
        exo_features = pd.concat([exo_features, cyclical_features], axis=1)
        print(exo_features.head)

    def search_space(self, trial):
        search_space  = {
            'n_estimators'  : trial.suggest_int('n_estimators', 400, 1200, step=100),
            'max_depth'     : trial.suggest_int('max_depth', 3, 10, step=1),
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.5),
            'reg_alpha'     : trial.suggest_float('reg_alpha', 0, 1, step=0.1),
            'reg_lambda'    : trial.suggest_float('reg_lambda', 0, 1, step=0.1),
        } 
        return search_space
    
    def train_model(self):
        forecast = ForecasterAutoreg(regressor = XGBRegressor(random_state = 1543),lags = 10)
        forecast.fit(y=trainy)
        results_search, frozen_trial = bayesian_search_forecaster(
                                        forecaster         = forecast,
                                        #y                  = df3.loc[:end_validation, "total actual load"],
                                        #exog               = df3.loc[:end_validation, df3.columns!="total actual load"],
                                        y                  = temp_features.loc[:end_validation, "total load actual"],
                                        exog               = temp_features.loc[:end_validation, temp_features.columns!="total load actual"],
                                        search_space       = search_space,
                                        lags_grid          = lags_grid,
                                        steps              = 36,
                                        refit              = False,
                                        metric             = 'mean_absolute_error',
                                        initial_train_size = len(trainy),
                                        fixed_train_size   = False,
                                        n_trials           = 20,
                                        random_state       = 123,
                                        return_best        = True,
                                        n_jobs             = 'auto',
                                        verbose            = False,
                                        show_progress      = True
                                    )


        
model1 = xgboost_model()
model1.train_model()








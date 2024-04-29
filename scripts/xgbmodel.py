from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import data 
#from skopt.space import Real, Integer
#from skopt import BayesSearchCVh
import pandas as pd
from joblib import dump, load
import numpy as np
import pickle
import os

from skforecast.ForecasterBaseline import ForecasterEquivalentDate
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import check_y


from sklearn.preprocessing import PolynomialFeatures


class xgboost_model():
    def __init__(self,dataframe):
        self.df = dataframe
        #self.df = data.preprocessing(self.df)
        #self.trainx,self.testx,self.trainy,self.testy, self.end_validation = data.split_data(self.df,"Actual Load","2023-03-01 23:00:00","2023-06-30 23:00:00")

    def split_data(self):
        self.trainx, self.testx, self.trainy, self.testy,self.end_validation = data.split_data(self.df,"total load actual","2023-03-01 23:00:00","2023-06-30 23:00:00")

    def build_pipeline(self):
        estimators = [
            ("reg", XGBRegressor())
        ]

        return Pipeline(steps = estimators)

    def create_features(self,weather_df):
        
        self.exo = data.create_features(self.df,weather_df)
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
            'temp_roll_mean_1_day',
            'temp_roll_mean_7_day',
            'temp_roll_max_1_day',
            'temp_roll_min_1_day',
            'temp_roll_max_7_day',
            'temp_roll_min_7_day',
            'temp'

        ]

        poly_features = transformer_poly.fit_transform(self.exo[poly_cols].dropna())
        poly_features = poly_features.drop(columns=poly_cols)
        poly_features.columns = [f"poly_{col}" for col in poly_features.columns]
        poly_features.columns = poly_features.columns.str.replace(" ", "__")

        self.exo = pd.concat([self.exo, poly_features], axis=1)
        
        

    def search_space(self, trial):
        self.search_space  = {
            'n_estimators'  : trial.suggest_int('n_estimators', 400, 1200, step=100),
            'max_depth'     : trial.suggest_int('max_depth', 3, 10, step=1),
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.5),
            'reg_alpha'     : trial.suggest_float('reg_alpha', 0, 1, step=0.1),
            'reg_lambda'    : trial.suggest_float('reg_lambda', 0, 1, step=0.1),
        } 
        return self.search_space
    
    def train(self):
        self.exo = data.create_features(self.df)
        self.exo = self.exo.dropna()
        print(self.exo.head)
        self.df = pd.concat([self.df["total load actual"],self.exo],axis = 1)
        #print(self.df.head)

    def train_models(self,weather):
        
        #self.create_features()

        self.exo = data.create_features(self.df,weather)

        features = []
        # Columns that ends with _sin or _cos are selected
        features.extend(self.exo.filter(regex='^sin_|^cos_').columns.tolist())
        # columns that start with temp_ are selected
        features.extend(self.exo.filter(regex='^temp_.*').columns.tolist())
        # Columns that start with holiday_ are selected
        features.extend(['temp'])
        self.exo = self.exo.filter(features, axis=1)

        #removes temp features for testing
        features = [x for x in features if "temp" not in x]
        print(features)
        """
        self.df = self.df[["total load actual"]].merge(
                self.exo,
                left_index=True,
                right_index=True,
                how='left'
            )"""

        self.df = pd.concat([self.df["total load actual"],self.exo],axis = 1)
        #print(self.exo.index)
        #self.exo = self.exo.dropna()
        #print(self.exo.index)
        
        self.df = self.df.dropna()
        self.split_data()

        #print(self.df.iloc[0])
        print(self.df.head)
        #print(self.df.index)
        
               
        self.forecast = ForecasterAutoreg(regressor = XGBRegressor(random_state = 1543),lags = 168)
        self.forecast.fit(y=self.trainy)
        lags_grid = [[1, 2, 3, 23, 24, 25, 167, 168, 169]]
        results_search, frozen_trial = bayesian_search_forecaster(
                                        forecaster         = self.forecast,
                                        #y                  = df3.loc[:end_validation, "total actual load"],
                                        #exog               = df3.loc[:end_validation, df3.columns!="total actual load"],
                                        y                  = self.df.loc[:self.end_validation, "total load actual"],
                                        #temp_features.columns!="total load actual"
                                        exog               = self.df.loc[:self.end_validation, features],
                                        search_space       = self.search_space,
                                        lags_grid          = lags_grid,
                                        steps              = 24,
                                        refit              = False,
                                        metric             = 'mean_absolute_error',
                                        initial_train_size = len(self.trainy),
                                        fixed_train_size   = False,
                                        n_trials           = 20,
                                        random_state       = 123,
                                        return_best        = True,
                                        n_jobs             = 'auto',
                                        verbose            = False,
                                        show_progress      = True
                                    )
       


    def backtesting(self):
        metric, predictions = backtesting_forecaster(
                          forecaster         = self.forecast,
                          y                  = self.df["total load actual"],
                          steps              = 24,
                          metric             = 'mean_absolute_error',
                          initial_train_size = len(self.trainy),
                          refit              = False,
                          n_jobs             = 'auto',
                          verbose            = True, # Change to False to see less information
                          show_progress      = True
                      )
        print(metric)

    

    def save_model(self,filename):
        """        
        check_file = os.path.isfile(f"models/{filename}")
        if not check_file:
            f = open("myfile.txt", "w")"""
        pickle.dump(self.forecast,open(f"models/{filename}","wb"))
      
df = data.load_data(dataset="datasets/energy_updated.csv")    
df = data.preprocessing(df) 


model1 = xgboost_model(df)
weather_df = data.weather_data()
weather_df = weather_df.loc[df.index[0]:df.index[-1],:] 

model1.train_models(weather_df)

model1.backtesting()

model1.save_model("xgboost_v1_no_temp.joblib")
"""
df2 = data.load_data()
df2 = data.preprocessing(df2)

model2 = xgboost_model(df2)
weather_df = pd.read_csv("../datasets/weather_features.csv")
weather_df = weather_df.loc[weather_df["city_name"] == "Madrid"]
weather_df = weather_df.drop_duplicates(subset="dt_iso")
weather_df = weather_df.loc[df2.index[0]:df2.index[-1],:] 
model2.train_models(weather_df)
"""

"""
df_missing = weather_df[weather_df.index.diff()>pd.Timedelta('1H')]
df_missing['diff'] = weather_df.diff()
print(df_missing)
"""
#model1.create_features(weather_df)
#print(model1.exo.head)
#model1.train()








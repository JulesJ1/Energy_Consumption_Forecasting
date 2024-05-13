import data 
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import pickle


class xgboost_model():
    """
    Class to create an XGBoost model

    Attributes:
        dataframe: Input dataframe used to train and test the model.
        target: Name of the dataframe column that will be used as the target.
        training_set: Timestamp of the end of the training set.
        test_set: Timestamp of the beginning of the test set.
    
    
    """

    def __init__(self,dataframe,target,train_set,test_set):
        """
        Constructs the attributes of the XgBoost model.
        """
        self.df = dataframe
        self.target,self.train_set,self.test_set = target,train_set,test_set


    def split_data(self):
        """
        Splits the daframe into training, validation and test sets.
        """

        self.trainx, self.testx, self.trainy, self.testy,self.end_validation = data.split_data(self.df,self.target,self.train_set,self.test_set)

    def build_pipeline(self):
        """
        Builds the model pipeline.
        """

        estimators = [
            ("reg", XGBRegressor())
        ]

        return Pipeline(steps = estimators)
        

    def search_space(self, trial):
        """
        The search space provides tuning parameters for the skforecast autoregressor using optuna trials.
        """

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


    def train_models(self,pols, weather = None):
        """
        Trains an XGBoost model using skforecast including a grid of lag values, exogenous variables and tuning parameters.

        Args:
            pols: A list of strings used to name polynomial exogenous variables used in the model training.
            weather: Creates rolling temperature average, minimum and maximum values as exogenous variables.
        
        """
        
        if weather:
            self.exo = data.create_features(self.df,pols,weather)
        else:
            self.exo = data.create_features(self.df,pols)


        # Select the names of the features to be used as exogenous variables during training
        features = []
        
        features.extend(self.exo.filter(regex='^sin_|^cos_').columns.tolist())
        
        features.extend(self.exo.filter(regex='^temp_.*').columns.tolist())
        
        features.extend(['temp'])
        self.exo = self.exo.filter(features, axis=1)

        features = [x for x in features if "temp" not in x]

        self.df = pd.concat([self.df["total load actual"],self.exo],axis = 1)
        
        self.df = self.df.dropna()
        self.split_data()
                     
        self.forecast = ForecasterAutoreg(regressor = XGBRegressor(random_state = 1543),lags = 168)
        self.forecast.fit(y=self.trainy)
        lags_grid = [[1, 2, 3, 23, 24, 25, 167, 168, 169]]
        results_search, frozen_trial = bayesian_search_forecaster(
                                        forecaster         = self.forecast,
                                        y                  = self.df.loc[:self.end_validation, "total load actual"],
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
                                        show_progress      = True,
                                    )
       


    def backtesting(self):
        """
        Backtests the created model on the test set and returns the mean absolute error of the model.
        """
        metric, predictions = backtesting_forecaster(
                          forecaster         = self.forecast,
                          y                  = self.df["total load actual"],
                          steps              = 24,
                          metric             = 'mean_absolute_error',
                          initial_train_size = len(self.trainy),
                          refit              = False,
                          n_jobs             = 'auto',
                          verbose            = True, 
                          show_progress      = True
                      )
        print(metric)

    

    def save_model(self,filename):
        """
        Saves the model as ajoblib file in the models folder.
        
        Args:
            filename: The name that the model will be saved as.
        """
        pickle.dump(self.forecast,open(f"models/{filename}","wb"))
      








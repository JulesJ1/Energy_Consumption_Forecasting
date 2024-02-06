from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import data 
from skopt.space import Real, Integer
from skopt import BayesSearchCV
import pandas as pd




class xgboost_model():
    def __init__(self):
        self.df = data.load_data()

    def split_data(self):
        self.trainx, self.testx, self.trainy, self.testy = data.split_data(self.df)

    def build_pipeline(self):
        estimators = [
            ("reg", XGBRegressor())
        ]

        return Pipeline(steps = estimators)

    def parameter_tuning(self):
        search_params = {
            "reg__max_depth": Integer(2,8),
            "reg__learning_rate": Real(0.001,0.1,prior="log-uniform"),
            "reg__subsample": Real(0.5,1.0),
            "reg__reg_alpha": Real(0.0,10.0),
            "reg__reg_lambda": Real(0.0,10.0),
            "reg__gamma": Real(0.0,10.0)

        }
        pipe = self.build_pipeline()
        opt = BayesSearchCV(pipe, search_params,cv=2,n_iter = 10, scoring = "r2")

        return opt
    
    def train_model(self):
        self.split_data()
        optimizer = self.parameter_tuning()
        optimizer.fit(self.trainx,self.trainy)

        print(optimizer.best_score_)
    


        
model1 = xgboost_model()
model1.train_model()








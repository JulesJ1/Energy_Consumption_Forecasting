from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import data 


class xgboost_model():
    def __init__(self):
        self.df = data.load_data()
        self.trainx, self.testx, self.trainy, self.testy 

    def split_data(self):
        self.trainx, self.testx, self.trainy, self.testy = data.testtrainsplit(self.df)

    def build_pipeline(self):
        estimators = [
            ("reg", XGBRegressor())
        ]

        pipeline = Pipeline(steps = estimators)

    def parameter_tuning(self):
        search_params = {
            "reg__max_depth"
            "reg__learning_rate"
            "reg__subsample"
            "reg__reg_alpha"
            "reg__reg_lambda"
            "reg__gamma"

        }

        



#print(df.head())





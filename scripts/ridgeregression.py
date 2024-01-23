import matplotlib.pyplot as plt
from numpy import array
import numpy as np  
import pandas as pd
import os
import random
import math
""" 
Ridge Regression Class From Scratch

"""
class Ridge_Regression():
    def __init__(self,file,size):
        
        self.file = file    
        self.size = size
        
        self.X_train = []
        self.X_test = [] 
        self.y_train = []
        self.y_test = []
        self.time = []
        self.R2 = 0 
        
    """ 
    Function opens csv file and formats the date column using pandas so it is in year-month-day format
    """
    def openfile(self):
        self.rf = pd.read_csv(os.path.dirname(__file__) + self.file,nrows = self.size)


        self.rf.loc[:, "date"] = self.rf.loc[:, "date"].str.split(" ").str[0]  
        self.rf.loc[:, "date"] = pd.to_datetime(self.rf.loc[:, "date"], format="%Y-%m-%d")
        return self.rf
    
    """
    Normalizes the dataset using min-max normalization. Each feature ranges from 0 to 1
    """
    def nrmlze(self,col):
        
        minimum = 0 - col.min()
        for x in range(len(col)):
            col[x] += minimum
        
        maximum = col.max()
        for x in range(len(col)):
            col[x] = col[x] / maximum
        
        return col, minimum, maximum
    
    



    """
    
    Randomly splits the data into training and test subsets and applies normalization method

    """
    def split_data(self):
        self.openfile()
        
        sample = random.sample(range(self.size), self.size)  
        
        X = list(zip(self.rf["high"],self.rf["low"],self.rf["open"],self.rf["volume"]))

        y = self.rf["close"]
        for x in range(len(X)):
            X[x] = list(X[x])
            X[x].insert(0,1)


        #split data in chunks
        #           
        """
        for x in range(self.size):
            if x <= (self.size*0.25):

                self.X_train.append(X[sample[x]])
                self.y_train.append(y[sample[x]])
                self.time.append(self.rf["date"][x])
            else:
                self.X_test.append(X[sample[x]])
                self.y_test.append(y[sample[x]])

        """

        for x in range(self.size):
            if x <= (self.size*0.25):
                self.X_train.append(X[x])
                self.y_train.append(y[x])
                self.time.append(self.rf["date"][x])
            else:
                self.X_test.append(X[x])
                self.y_test.append(y[x])
        normx = np.array(self.X_train)
        print(normx[:,1])
       
        
        self.nrmlze(normx[:,1])
        for x in range(1,5):
            temp = self.nrmlze(normx[:,x] )
            for y in range(len(self.X_test)):
                 
                if y < len(self.X_train):
                    self.X_train[y][x] = temp[0][y]

                self.X_test[y][x] =(self.X_test[y][x] + temp[1])/ temp[2]

       



    
    """
    Calculates the predicted y-values and R^2 score for least square regression on the training data
    """
    def predictY(self, lmbda,X_train, y_train):
       
        I = np.identity(np.matmul(np.transpose(X_train)  ,X_train).shape[0])
        I[0][0] = 0
        lmbda = lmbda

        b0 = np.linalg.inv(np.matmul(np.transpose(X_train)  ,X_train) + lmbda * I)
        b0 = np.matmul(b0,np.matmul(np.transpose(X_train),y_train))
      

        pred_y = np.matmul(X_train,b0)  
   
        rss = ((y_train-pred_y)**2).sum()
        
        avg_y = sum(y_train)/len(y_train)

        tss = ((y_train-avg_y)**2).sum()

        R2 = (tss - rss )/ tss

        return pred_y , R2



    """
    
    Uses 5-fold Cross Validation to find the most suitable value for lambda.
    Lambda values tested = 0.01, 0.1, 1, 10, 100
    
    """
    
    def optimal_lmbda(self):

        
        values = [0.01,0.1,1,10,100]
        
        
       
        folds = [[],[],[],[],[]]
        for x in range(len(self.X_train)//5):
            index = len(self.X_train)//5
            folds[0].append(self.X_train[x])
            folds[1].append(self.X_train[x+index])
            folds[2].append(self.X_train[x + index*2] )
            folds[3].append(self.X_train[x + index*3])
            folds[4].append(self.X_train[x + index*4])

        
        splitX = [self.X_train[i::5] for i in range(5)]
        splity= [self.y_train[i::5] for i in range(5)]
        
        scores = [0,0,0,0,0]
        for x in range(len(values)):
            for y in range(5):
                temp = self.predictY(values[x],splitX[y],splity[y])
                scores[x] += temp[1]
        
        print(values[scores.index(max(scores))])
        return values[scores.index(max(scores))]
        

    """
    Plots the stock price times series training set data aswell as the predicted label values for the training set

    """

    def plot(self):

        self.split_data()
        lmbda = self.optimal_lmbda()
        predicted = self.predictY(lmbda,self.X_train,self.y_train)[0]
        l1 = plt.plot(self.time, self.y_train)
        plt.xlabel("Date")

        plt.ylabel("Price")
        l2 = plt.plot(self.time,predicted)
        plt.legend([l1,l2], ["Actual Value", "Predicted Value"])   
        plt.show()

         

         
    
if __name__ == '__main__':


    lsr = Ridge_Regression("/../prices.csv",250)
    lsr.plot()


         


    


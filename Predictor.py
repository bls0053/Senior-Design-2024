


import numpy as np
import pandas as pd

# Predictor object for pred_features
class FeaturePredictor:

    reduction = .1

    def __init__(self,
                 regressor,
                 target,
                 polarity = 1,
                 curr_val = 0,
                 weights = [],
                 lock = [],
                 allowed_error=.1, 
                 early_exit=100,
                 ):

        self.weights = weights
        self.curr_val = curr_val
        self.polarity = polarity
        self.regressor = regressor
        self.allowed_error = allowed_error
        self.early_exit = early_exit
        self.target = target
        self.lock = lock

    # Evaluates to TRUE if value is within Predictor margin of error
    def match(self):

        if (self.target - self.allowed_error <= self.curr_val <= 
            self.target + self.allowed_error):
            
            return True
        else:
            return False
        

    # Flags whether the current value needs to increase or decrease, 1 -> target is higher, -1 -> target is lower        
    def set_pol(self):

        if (self.target > self.curr_val):
            self.polarity = 1
        else:
            self.polarity = -1


    def init_weights(self, coef, df):

        ranges = df.apply(lambda x: x.max() - x.min())
        self.weights = np.zeros(df.shape[0])
        

        for i, row in enumerate(coef.index):
            self.weights[i] = (coef.iloc[i][coef.columns[0]])*(self.polarity)*(ranges.iloc[i])*self.reduction
            
            # Remove - for testing only
            print("Feature:", coef.index[i], "-----> "
                  "Coef:", coef.iloc[i][coef.columns[0]], 
                  " X Polarity:", self.polarity, 
                  " X Range:", ranges[i],
                  " = Weight:", self.weights[i], "\n")
            


    def modify_weights(coef, df):
        
        pass



    # Takes in single rowed dataframe, modulates feature values
    def stretch_feat(self, df):

        for i, column in enumerate(df.columns):

            # print(df.iloc[0][i])
            # print(self.weights[i])

            sum = float(df.iloc[0][i]) + self.weights[i]

            df[column] = df[column].replace(df.loc[df.index[0]][column], str(sum))

        return df



    def pred():
        pass





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
                 allowed_error='.1', 
                 early_exit='100',
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

        print(coef)

        print(ranges)

        for i, coefficient in enumerate(coef):

            self.weights[i] = (ranges.iloc[i])*(float(coefficient))*(self.reduction)


    def modify_weights(coef, df):
        
        pass










    # Takes in single rowed dataframe, modulates feature values
    def stretch_feat():
        pass

import lazypredict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from IPython.display import display
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize

from clean import preproc, init_df, format
from models import lasso_cv, lz_reg
import re



##################################################################################################################

# data_init -> Starting dataset, contains missing values and categorical data
# data_clean -> Data after preprocessing, missing values removed or filled with column mean, categorical removed
# df_ut -> Lasso on un-tuned parameters
# df_t -> Lasso with parameter tuning
# df_t_coef -> Sorted list of parameter tuned Lasso coefficients




data_init = pd.read_csv('AL_Dist.csv')

data_init, features = init_df(data_init)

inp = input("Welcome Prof. Pendola, press any button to start\n")

while(inp != "Done"):

    for i, group in enumerate(features):
        if (i==0):
            print("Categorical Variables(Specify desired choices):")
        if (i==1):
            print("Numerical Variables (Specify desired percent of student body):")
        for feat in group:
            print(feat.name.capitalize())


    inp = input("\nChoose feature to modify (Type 'Done' when finished): ")

    for i, group in enumerate(features):
        for j, feat in enumerate(group):

            if (inp.casefold() == feat.name):
                obj = features[i][j]
                name = obj.name
                data = obj.data

                if i == 0:

                    print("\nValues: ", data, "\n")
                    inp = input("Choose variables to include: ")
                    inp = format(inp)

                    

                    mask = data_init[name].isin(inp)
                    data_init = data_init[mask]

                    display(data_init)
                    

                else:
                    
                    inp = input("\nChoose percentage (value between 0-1): ")
                    data_init = data_init[data_init[name] >= float(inp)]

                    display(data_init)

                inp = ""
                    





# data_clean = preproc(data_init)
# df_ut, df_t, df_t_coef = lasso_cv(data_clean)
# reg_models = lz_reg(data_clean)









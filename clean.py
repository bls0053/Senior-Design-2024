import lazypredict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from IPython.display import display
import numpy as np

# from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
import re


##########################################################################################################################################################################

def preproc(df):

    # Remove unecessary/gappy and categorical data (columns)
    drop_cols = [
        'leaid',
        'leanm',
        'achv',
        'CT_EconType',
        'DIST_FACTORS',
        'COUNTY_FACTORS',
        'HEALTH_FACTORS',
        'H_FCTR_ZS',
        'LOCALE_VARS',
        'Locale4',
        'Locale3',
        'math',
        'rla'
    ]
    df.drop(columns=drop_cols, inplace=True)

    # Remove gappy rows / Make tool in future to find gaps
    drop_rows = [
        688, 724, 726, 3422, 3874, 4243, 4244, 4249, 4250, 4256, 4257, 4261, 4262, 4618
    ]
    df.drop(index=drop_rows, inplace=True)

    # Remove gappy rows in range
    mask = (df.index < 4781) | (df.index > 4788)
    df = df[mask]

    mask = (df.index < 4662) | (df.index > 4729)
    df = df[mask]

    mask = (df.index < 490) | (df.index > 553)
    df = df[mask]
    
    # Replace missing values with mean
    for column in df.columns:

        if df[column].isnull().any():
             
             mean = df[column].mean()
             df[column].fillna(mean, inplace=True)

    return df

##########################################################################################################################################################################





class Feature_c:
    def __init__(self, name, data):
        self.name = name
        self.data = data

class Feature_n:
    def __init__(self, name, data):
        self.name = name
        self.data = data


# Categorical variables
cat_var = ['grade', 'year', 'leanm', 'Locale4', 'Locale3', 'BlackBeltSm', 'FoodDesert', 'CT_EconType']
cat_str = ['leanm','Locale4','Locale3','CT_EconType']

# Numerical variables, cannot select for more than 100% in subset
num_var = ['perasn','perblk','perwht','perind','perhsp', 'perecd', 'perell']




def init_df(df):

    for i, var in enumerate(cat_var):
        name = cat_var[i]
        cat_var[i] = Feature_c(name, df[name].unique())
    
    for j, var in enumerate(num_var):
        name = num_var[j]
        num_var[j] = Feature_n(name, df[name].unique())

    features = [cat_var, num_var]
    
    for k in cat_str:
        df[k] = df[k].str.lower()

    
    display(df)

    return df, features



def format(str):
    str = str.split(',')
    formatted_array = []

    for i in str:
        if re.search(r'\d', i):
            formatted_element = int(i)
        else:    
            formatted_element = i.lower()
        formatted_array.append(formatted_element)    

    return formatted_array

















# feat_list = ['locale4', 'grade', ]

# class Feature:
#     name = ''
#     data = []



# def subset(df, feat, range):

#     display(df)
#     print("\n", feat, "\n", range)

#     mask = df[feat].isin(range)
#     df = df[mask]

#     return df



# def subset(df, feat, range):

#     mask = df[feat].isin(range)
#     df = df[mask]



# Prompt user for Selection
# 	County/City
# 	Locale3/Locale4
# 	Race
# 	Grade
# 	Econ Type
# 	Custom?
# 	Done
#   Reset
#       >Confirm
#       >Back ---^
#
# 	For chosen 
# 		Shows choices
# 			Ex. 'grade'
# 			Choose: "1, 2, 3, 4, 5, 6, 7, 8"
# 				Select 1 or multiple (comma delimited?) -i - e
# 				> 1,2,3
#               > 8
#               > confirm
#               >Back ---^
#   go back to selection
#   remove options already chosen
#
# Prompt user for Selection
# 	County/City
# 	Locale3/Locale4
# 	Race
# 	Econ Type]
# 	Custom?
# 	Done
#   Reset
#       >Confirm
#       >Back ---^
#
#   Etc
#
#   Subset() function takes in array, removes from dataframe all but input features
#       ex. [1,2,3,4]
#           remove 5,6,7,8
#
#   create feature object 
#       - row vs col
#       - range of acceptable values?







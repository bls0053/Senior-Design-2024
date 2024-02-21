

# Imports
import numpy as np
from IPython.display import display
import re



############################################################################### Dataframe Init ###############################################################################

# Class object for categorical features
class Feature_c:
    def __init__(self, name, data):
        self.name = name
        self.data = data

# Class object for numerical features
class Feature_n:
    def __init__(self, name, data):
        self.name = name
        self.data = 0


# Categorical variables
cat_var = ['grade', 'year', 'leanm', 'Locale4', 'BlackBeltSm', 'FoodDesert', 'CT_EconType']

# Nominal variables
cat_str = ['leanm','Locale4','CT_EconType']

# Numerical variables, cannot select for more than 100% in subset
num_var = ['perasn','perblk','perwht','perind','perhsp', 'perecd', 'perell']


# Initializes changeable features for subset creation
    # Refactor - make single loop, reduce redundancy
def init_df(df):

    temp_df = df.copy()
    # temp_df.set_index('leaid')

    # Always drop
        # leaid -> noise, achv -> alternate of predicted metric, Locale3 -> alternate of Locale4, math -> 1 to 1 of achvz, rla -> 1 to 1 of achvz
    drop_cols = [
        'leaid', 
        'achv', 
        'Locale3', 
        'math', 
        'rla'
    ]

    temp_df.drop(columns=drop_cols, inplace=True)



    # Drop columns if they are fully empty. Should be: LOCALE_VARS, DIST_FACTORS, HEALTH_FACTORS, COUNTY_FACTORS
    for column in temp_df.columns:
        if ~temp_df[column].notna().any():
            temp_df.drop(columns=column, inplace=True)


    for i, var in enumerate(cat_var):
        name = cat_var[i]

        data = np.array(temp_df[name].unique())
        data = remove_nan(data)

        cat_var[i] = Feature_c(name, data)
        
    for j, var in enumerate(num_var):
        name = num_var[j]
        
        num_var[j] = Feature_n(name, data)

    
    features = [cat_var, num_var]
    
    for k in cat_str:
        temp_df[k] = temp_df[k].str.lower()

    
    display(temp_df)

    return temp_df, features


# Formats input - Removes whitespace, ignores case, converts type
def format(str):
    str = str.split(',')
    formatted_array = []

    for i in str:
        if re.search(r'\d', i):
            formatted_element = int(i)
        else:    
            formatted_element = i.lower().strip()
        formatted_array.append(formatted_element)    

    return formatted_array


# Removes 'nan' values in feature data
def remove_nan(arr):
     return np.array([item for item in arr if str(item).lower() != 'nan'])




############################################################################### Cleaning ############################################################################### 

# Two methods per issue
    # Nominal values - Either drop them or encode them
    # Empty values - Either drop them or populate them

# Binary encoding for categorical variables
def bin_encode(df):
    pass



# Drop nominal values - Extremely sloppy, refactor in the future
def drop_nom(df):
    for column in df.columns:
        if df[column].astype(str).str.contains(r'[0-9.-]', regex=True).any():
            pass
        else:
            df.drop(columns=column, inplace=True)


# Fill nan values
def mean_sub(df):
    for column in df.columns:

        if df[column].isnull().any():
             
            mean = df[column].mean()
            df[column].fillna(mean, inplace=True)


# Drop nan values in groups, hinge on specified/default parameter
def drop_gap(df):
    pass


# Print current subset feature object data
def prt_feat_data(features):
    for i, group in enumerate(features):
        for j, feat in enumerate(group):
            name = feat.name
            data = feat.data
            print(f"Feature Name: {name}\tFeature Data: {data}\n")






# Misc Notes


# priorities when coming back, refactor pre-processing method - fill missing values: ALL? or minor?
# remove big chunks before subset selection - static/init "drop_gaps()", "df_init()"
# then mean substitution for remaining holes - after/dependent "mean_sub()"
# Preproc() -> df_init(), ____Subset____, mean_sub() (and encoding), Models, graphs
# encode
# ct_econtype
# Locale4
# leanm
#
# New prios: 
# encode nominals
# drop gappy?
# Integrate graphing / refactor graphing
#
#
#
#
#
# When only using 1 value of nominal value, can drop column, otherwise binary encode 

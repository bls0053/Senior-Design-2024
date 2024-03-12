


# Imports
import pandas as pd
from IPython.display import display

from clean import init_df, format, prt_feat_data, drop_nom, mean_sub
import models
from Predictor import FeaturePredictor
import numpy as np

from tqdm import tqdm
import time


##################################################################################################################

# data_init -> Starting dataset (df)
# data_subset -> Chosen data subset (df)
# features -> List of Feature objects ([][])
# lasso_metrics -> results from lasso (df)
# lasso_coef -> coefs from lasso (df)
# lasso_coef_red -> coefs reduced down (df)
# lzp_metrics -> 


# Initializing dataframe and changeable features
data_init = pd.read_csv('AL_Dist.csv')
data_subset, features = init_df(data_init)


############################################################################### Subset Creation ###############################################################################
#
# Need to add:
    # Make 'done' case-insensitive
    # Add reset option
    # Warning for the dataset becoming too small upon making a choice
    #   or: give an option to revert choice
    # If mispelled print "not a viable option, try again"
    #

inp = input("Welcome Prof. Pendola, press Enter to start\n")

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
            if (inp.lower() == feat.name.lower()):

                obj = features[i][j]
                name = obj.name
                data = obj.data

                if i == 0:

                    print("\nValues: ", data, "\n")
                    inp = input("Choose variables to include: ")
                    inp = format(inp)
                    obj.data = inp

                    
                    print(inp)
                    mask = data_subset[name].isin(inp)
                    data_subset = data_subset[mask]

                    # print(data_subset[1:400].to_string())
                    display(data_subset)
                    

                else:
                    
                    inp = input("\nChoose percentage (value between 0-1): ")
                    obj.data = inp

                    data_subset = data_subset[data_subset[name] >= float(inp)]

                    # print(data_subset[1:400].to_string())
                    display(data_subset)

                inp = ""

############################################################################### Preprocessing/One-hot ###############################################################################
#
# Need to add?:                
    # Tune lasso parameters proportionally to dataset size***
    # Prompt user input for what they want to see               
    # Integrate graphs
    # Optional choice for user to hard-change params
    # 




# Prompt user for options
#   Look at data
#       Full data
#       Subset
#       Reduced Subset
#       Lasso coefs
#       Reduced coefs
#   Manipulate data   
#       drop any
#       drom nom
#       new subset
#   Models
#       Run lasso on ()
#       Run lazypredict on ()
#       Train model on ()
#       Run model on ()
#           predict achvz
#           predict features
# 

drop_nom(data_subset)
mean_sub(data_subset)


lasso_metrics, lasso_coef = models.lasso_cv(data_subset)

subset_red, coef_red = models.reduce_subset(data_subset, lasso_coef, 0.1)


# lzp_metrics = models.lz_reg(data_subset)
# new_models = models.get_models(lzp_metrics)


display(data_init, "\n")
display(data_subset, "\n")


display(lasso_metrics, "\n")
display(lasso_coef, "\n")
display(coef_red, "\n")
display(subset_red, "\n")

# display(lzp_metrics, "\n")
# display(new_models, "\n")

prt_feat_data(features)












############################################################################### Achvz Predict - WIP ###############################################################################




inp = input("Feature Prediction, press Enter to start\n")


print(coef_red)
print(subset_red)
mod = models.ext_trees(subset_red)

# Placeholder for initializing model used for Predictor
tree_regressor = mod
pred = FeaturePredictor(regressor=tree_regressor, target=.95)

# Initializes starting feature weights / value to be changed
x = subset_red.drop('achvz', axis=1)
pred.init_weights(coef_red, x)

# Placeholder for selcted row to predict
pred_row = subset_red.iloc[0:1,0:]

# Creates variable for starting 'achvz'
curr_achvz = pred_row.loc[pred_row.index[0]]['achvz']
pred.curr_val = curr_achvz

# Initializes early exit count to 0
ee_count = 0

# Original row
x_row = pred_row.drop('achvz', axis=1)

# Modified row
mod_x_row = x_row.copy()
# pred.init_lock(mod_x_row)


############################################################################### Change Lock ###############################################################################

print(pred.lock, "\n")
inp = format(input("Modify Locks: "))

for i in inp:
    pred.lock.iloc[3][int(i)] = 0
inp = input()
###########################################################################################################################################################################


all = []
num_iterations = 100

while((pred.match() == False) and (ee_count < pred.early_exit)):

    # Set pol and stretch features
    mod_x_row = pred.stretch_feat(mod_x_row)

    # Predict modified achvz
    for i in tqdm(range(num_iterations), desc="Chuggin:", unit="iteration"):
        prediction = mod.predict(mod_x_row)
        all.append(prediction)
    mean_predictions = np.mean(all, axis=0)

    # Set new achvz progress
    pred.curr_val = mean_predictions
    pred.set_pol()
    
    # Iterate early exit count
    ee_count += 1

    print("polarity = ", pred.polarity)
    print("trial # = ", ee_count, " / ", pred.early_exit)
    print("Modified row = \n", mod_x_row)
    print("Weights = \n", pred.lock)
    print("Achvz = ", pred.curr_val, " / ", pred.target, "\n")
    
















# Todo
    # Implement Lock
    # Change reduction

    # Establish bounds
    # Collect differences in row
    
    # Graph 
    # 
    # 
    # 
    # 
    # 
    # 
    # 









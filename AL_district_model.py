
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

from clean import preproc, init_df, format, bin_encode, prt_feat_data, drop_nom, mean_sub
from models import lasso_cv, lz_reg, get_models, reduce_subset, ext_trees
import re




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
# Tune lasso parameters proportionally to dataset size***
# Prompt user input for what they want to see               
# Integrate graphs
# Optional choice for user to hard-change params
#                



drop_nom(data_subset)
mean_sub(data_subset)


lasso_metrics, lasso_coef = lasso_cv(data_subset)

subset_red, coef_red = reduce_subset(data_subset, lasso_coef)


# lzp_metrics = lz_reg(data_subset)
# new_models = get_models(lzp_metrics)


tree_df = ext_trees(subset_red)
display(tree_df)

# display(data_init, "\n")
# display(data_subset, "\n")
# display(lasso_metrics, "\n")
# display(lasso_coef, "\n")
# display(coef_red, "\n")
# display(subset_red, "\n")
# display(lzp_metrics, "\n")
# display(new_models, "\n")

# prt_feat_data(features, "\n")







# run trees without lasso, then run trees with























# lasso_cv_metrics, lasso_cv_coefs =  lasso_cv(data_init)













# Modules: gen_table() -> for misc metrics, list data, gen_barhplot() -> lazypredict results, lassocv coefs



# Prompt user for model choice
# Lasso, LassoCv, Lazypredict/lazyregressor, Lassocv with range of coefficients selected, lassocv with coefficients removed then run on new models 1 of 3-5
# graph
#
#
#
#
#
#
#
#
#
#                
#
#
#
#
#
#
#
#                
#
#
#
#                
# data_clean = preproc(data_init)
# df_ut, df_t, df_t_coef = lasso_cv(data_clean)
# reg_models = lz_reg(data_clean)









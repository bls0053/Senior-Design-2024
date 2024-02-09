


# Imports
import pandas as pd
from IPython.display import display

from clean import init_df, format, prt_feat_data, drop_nom, mean_sub
import models




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



drop_nom(data_subset)
mean_sub(data_subset)


lasso_metrics, lasso_coef = models.lasso_cv(data_subset)

subset_red, coef_red = models.reduce_subset(data_subset, lasso_coef, 0.001)


lzp_metrics = models.lz_reg(data_subset)
new_models = models.get_models(lzp_metrics)


display(data_init, "\n")
display(data_subset, "\n")


display(lasso_metrics, "\n")
display(lasso_coef, "\n")
display(coef_red, "\n")
display(subset_red, "\n")

display(lzp_metrics, "\n")
display(new_models, "\n")

prt_feat_data(features)



print("\n")
models.ext_trees(subset_red)
models.grad_boost(subset_red)
models.nu_svr(subset_red)
models.svr(subset_red)
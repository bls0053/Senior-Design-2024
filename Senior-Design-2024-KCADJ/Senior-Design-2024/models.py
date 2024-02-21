
# Imports
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from IPython.display import display
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.utils._testing import ignore_warnings

from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import NuSVR, SVR
import lightgbm as ltb



# Lasso
    # 0.2/0.8 split (test=.2)
    # Gridsearch tunes param (alpha, tol), alpha may be hardset in future
    # Returns feature coefficient strengths and other Lasso performance metrics

def lasso_cv(df):

    # Set target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # train_test_split: test=.2, train=.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # Setup Lasso
    lasso = Lasso()
    lasso.fit(X_train, y_train)

    # Lasso cross validation w/ tuning 
    param_grid = {
        'alpha' : [0.01, 0.1, 1, 10, 100],
        'tol' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    }
    lasso_cv = GridSearchCV(lasso, param_grid, cv = 5, n_jobs = -1)
    lasso_cv.fit(X_train, y_train)
    y_pred2 = lasso_cv.predict(X_test)
    
    # Results
    lasso2 = lasso_cv.best_estimator_
    lasso2.fit(X_train, y_train)

    df_t = pd.DataFrame(columns = ["Mean absolute Error", "Mean Squared Error", "R2 Score", "Lasso Vars"])
    values = [mean_absolute_error(y_test, y_pred2), mean_squared_error(y_test, y_pred2), r2_score(y_test, y_pred2), lasso_cv.best_estimator_]
    df_t.loc[0] = values

    feature_names = df.columns.tolist()
    feature_names.remove('achvz')

    df_t_coef = pd.DataFrame(lasso2.coef_, columns =['Coefficients'], index=feature_names)
    
        # Reintroduce in seperate sort method
    # df_t_coef_sorted = df_t_coef.iloc[np.argsort(np.abs(df_t_coef['Coefficients']))]

    return df_t, df_t_coef


# Runs lazypredict on user-made dataframe. Returns dataframe populated with model performance results
def lz_reg(df):

    # Set target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']
    normalize(X)

    # train_test_split: test=.2, train=.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    reg_models, reg_predictions = reg.fit(X_train, X_test, y_train, y_test)
    reg_models = reg_models.sort_values(by='RMSE', ascending=True)


    return reg_models


# Returns the top n performing models
def get_models(df, amount = 3):

    temp_df = df.copy()
    temp_df = temp_df[0:amount]

    return temp_df



# Remove in future potentially. Same as reduce_subset but only reduces coefficient dataframe

    # def reduce_coef(df, reduc = .05):

    #     temp_df = df.copy()

    #     for row in temp_df.index:
    #         if np.abs(temp_df.loc[row,'Coefficients']) < reduc:
    #             temp_df.drop(index=row, inplace=True)

    #     return temp_df




# Reduces subset and coefficients based on specified amount of regularization of coefficients
def reduce_subset(df_sub, df_coef, reduc = .05):

    temp_sub = df_sub.copy()
    temp_coef = df_coef.copy()

    for row in df_coef.index:
        if np.abs(df_coef.loc[row,'Coefficients']) < reduc:
            temp_sub.drop(columns=row, inplace=True)
            temp_coef.drop(index=row, inplace=True)

    return temp_sub, temp_coef




# Simple implementation for ExtraTreesRegressor
def ext_trees(df):
    
    # Get target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # 80/20 test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Fit model
    regressor = ExtraTreesRegressor(n_estimators=200)
    regressor.fit(X_train, y_train)

        
    # Tune params - Way Way Way too heavy on execution time
    
        # param_grid = {
        #     'n_estimators' : [250, 500, 1000],
        #     'max_depth' : [None, 10, 20, 30]
        # }

        # regressor_t = GridSearchCV(regressor, param_grid, cv = 5, n_jobs = -1)
        # regressor_t.fit(X_train, y_train)
        # print(regressor_t.best_estimator_)
        # regressor2 = regressor_t.best_estimator_
    

    # Cross-val
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=None)
    n_scores = cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')


        # Reintroduce later in separate function
    # print('ExtraTreesRegressor MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    y_pred = regressor.predict(X_test)
    
    y_test = np.array(y_test)

    # act_v_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    return regressor


# Simple implementation for HistGradientBoostingRegressor
def grad_boost(df):
     
    # Get target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # 80/20 test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Fit model
    regressor = HistGradientBoostingRegressor()
    regressor.fit(X_train, y_train)

    # Cross-val
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=64)
    n_scores = cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

    print('HistGradientBoosting MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores))) 


def svr(df):
    # Get target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # 80/20 test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Fit model
    regressor = SVR()
    regressor.fit(X_train, y_train)

    # Cross-val
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=64)
    n_scores = cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

    print('SVR MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores))) 


# Simple implementation for LGBMRegressor. DOES NOT WORK
def lgbm(df):
    # Get target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # 80/20 test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Fit model
    regressor = ltb.LGBMRegressor()
    regressor.fit(X_train, y_train)

    # Cross-val
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=64)
    n_scores = cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

    print('LGBM MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores))) 


# Simple implementation for NuSVR
def nu_svr(df):
    # Get target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # 80/20 test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Fit model
    regressor = NuSVR()
    regressor.fit(X_train, y_train)

    # Cross-val
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=64)
    n_scores = cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

    print('NuSVR MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores))) 





# Takes in single dataframe element (row), model to be used, RETURNS - predicted achvz based off of features
def pred_achvz(df, mdl):
    pass


# Find range based weights for modification of feature prediction
# Take the coefs from reduced coef
# Find range of just those from full dataset
# mutiply it by some arbitrary number
# mutiply by weights later




# Takes in single dataframe element (row), model to be used, target achzv, locked variables, acceptable margin of error, RETURNS - predicted features for a target achvz
def pred_features(df, pred):

    # Creates variable for starting 'achvz'
    curr_achvz = df.loc[df.index[0]]['achvz']
    pred.curr_val = curr_achvz

    # Sets initial direction for predictor to go
    pred.set_pol()

    # Initializes early exit count to 0
    ee_count = 0

    while((pred.match(curr_achvz) == False) and (ee_count < pred.early_exit)):

        pred.stretch_feat(df)



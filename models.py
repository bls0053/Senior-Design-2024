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



# def lasso_cv(df):

#     # Set target and data
#     X = df.drop('achvz', axis=1)
#     y = df['achvz']
#     normalize(X)

#     # train_test_split: test=.2, train=.8
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

    
#     # Standardize data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.fit_transform(X_test)
    
#     # Run Lasso on un-tuned parameters
    
#     lasso = Lasso(tol=.00035)
#     lasso.fit(X_train, y_train)
#     y_pred = lasso.predict(X_test)
    
#     # Results
#     df_ut = pd.DataFrame(columns=["Mean Absolute Error", "Mean Squared Error", "R2 Score"])
#     values = [mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)]
#     df_ut.loc[0] = values
    

#     # Lasso cross validation w/ tuning
#     param_grid = {
#         'alpha' : [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#     }
#     lasso_cv = GridSearchCV(lasso, param_grid, cv = 3, n_jobs = -1)
#     lasso_cv.fit(X_train, y_train)
#     y_pred2 = lasso_cv.predict(X_test)
    

#     # Results
#     lasso2 = lasso_cv.best_estimator_
#     lasso2.fit(X_train, y_train)

#     df_t = pd.DataFrame(columns = ["Mean absolute Error", "Mean Squared Error", "R2 Score", "Lasso Vars"])
#     values = [mean_absolute_error(y_test, y_pred2), mean_squared_error(y_test, y_pred2), r2_score(y_test, y_pred2), lasso_cv.best_estimator_]
#     df_t.loc[0] = values

#     feature_names = df.columns.tolist()
#     feature_names.remove('achvz')

#     df_t_coef = pd.DataFrame({'Features': feature_names,
#                               'Coefficients': lasso2.coef_})
    

#     df_t_coef_sorted = df_t_coef.sort_values(by='Coefficients', ascending=False)
#     df_t_coef_sorted = df_t_coef.iloc[np.argsort(np.abs(df_t_coef['Coefficients']))]
    
#     return df_ut, df_t, df_t_coef_sorted


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
    # lasso = Lasso(tol=.00035)
    lasso = Lasso(tol=.0005)
    lasso.fit(X_train, y_train)

    # Lasso cross validation w/ tuning
    param_grid = {
        'alpha' : [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    }
    lasso_cv = GridSearchCV(lasso, param_grid, cv = 3, n_jobs = -1)
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

    df_t_coef = pd.DataFrame({'Features': feature_names,
                              'Coefficients': lasso2.coef_})

    # df_t_coef_sorted = df_t_coef.sort_values(by='Coefficients', ascending=False)
    df_t_coef_sorted = df_t_coef.iloc[np.argsort(np.abs(df_t_coef['Coefficients']))]

    return df_t, df_t_coef_sorted


def lz_reg(df):

    # regressors = [
    #     'ExtraTreesRegressor','LGBMRegressor','HistGradientBoostingRegressor','RandomForestRegressor','NuSVR','SVR','MLPRegressor',
    #     'KNeighborsRegressor','XGBRegressor','BaggingRegressor','GradientBoostingRegressor','AdaBoostRegressor','KernelRidge',
    #     'Ridge','RidgeCV','TransformedTargetRegressor','LinearRegression','BayesianRidge','LassoLarsCV','ElasticNetCV','LassoCV',
    #     'HuberRegressor','LassoLarsIC','SGDRegressor','LinearSVR','GaussianProcessRegressor','ExtraTreeRegressor','OrthogonalMatchingPursuitCV',
    #     'OrthogonalMatchingPursuit','LarsCV','TweedieRegressor','DecisionTreeRegressor','PassiveAggressiveRegressor','ElasticNet',
    #     'DummyRegressor','Lasso','LassoLars','RANSACRegressor'
    # ]

    # Set target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']
    normalize(X)

    # train_test_split: test=.2, train=.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    reg_models, reg_predictions = reg.fit(X_train, X_test, y_train, y_test)

    return reg_models

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



################################################## Modules #######################################################

# Import data
# Preprocessing / Cleaning
# LASSO Regression
# LazyClassifier
# LazyRegressor
# Data Visualization

##################################################################################################################



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
        'rla',
        # 'perblk',
        # 'perwht',
        # 'perecd'
    ]
    df.drop(columns=drop_cols, inplace=True)

    # Remove gappy rows / Make tool in future to find gaps
    drop_rows = [
        688, 724, 726, 3422, 3874, 4243, 4244, 4249, 4250, 4256, 4257, 4261, 4262, 4618
    ]
    df.drop(index=drop_rows, inplace=True)

    # Remove gappy rows in range
    # mask = (df.index < 490) | (df.index > 553 & df.index < 4662) | (df.index > 4729 & df.index < 4781) | (df.index > 4788)

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


def lasso_reg(df):

    # Set target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']
    normalize(X)

    # train_test_split: test=.2, train=.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # Run Lasso on un-tuned parameters
    
    lasso = Lasso(tol=.00035)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    
    # Results
    df_ut = pd.DataFrame(columns=["Mean Absolute Error", "Mean Squared Error", "R2 Score"])
    values = [mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)]
    df_ut.loc[0] = values
    

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
    

    df_t_coef_sorted = df_t_coef.sort_values(by='Coefficients', ascending=False)
    df_t_coef_sorted = df_t_coef.iloc[np.argsort(np.abs(df_t_coef['Coefficients']))]
    
    return df_ut, df_t, df_t_coef_sorted


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



##################################################################################################################

# data_init -> Starting dataset, contains missing values and categorical data
# data_clean -> Data after preprocessing, missing values removed or filled with column mean, categorical removed
# df_ut -> Lasso on un-tuned parameters
# df_t -> Lasso with parameter tuning
# df_t_coef -> Sorted list of parameter tuned Lasso coefficients

data_init = pd.read_csv('AL_Dist.csv')
data_clean = preproc(data_init)
df_ut, df_t, df_t_coef = lasso_reg(data_clean)
reg_models = lz_reg(data_clean)



fig, axes = plt.subplots(1,4, figsize=(20, 12))

axes[2].axis('off')
axes[3].axis('off')

################################################## Formatting for Lasso table #######################################################
ax = axes[3]

cell_text = [[str(val)] for val in df_ut.iloc[0]]
row_labels = df_ut.columns.to_list()

df_ut_table = ax.table(rowLabels=row_labels, cellText=cell_text, colWidths=[.8], 
                        bbox= [0.1, 0.89, 1.2, .1], colLabels=["Lasso"], rowColours=['#dce9fa', '#bfcad9', '#dce9fa', '#bfcad9'], 
                        colColours=['#bfcad9'], cellColours=[['#dce9fa'], ['#bfcad9'], ['#dce9fa']]
) 

df_ut_table.auto_set_font_size(False)
df_ut_table.set_fontsize(10)
df_ut_table.scale(1.1, 1.2)

################################################## Formatting for LassoCV table #######################################################

cell_text2 = [[str(val)] for val in df_t.iloc[0]]
row_labels2 = df_t.columns.to_list()

df_t_table = ax.table(rowLabels=row_labels2, cellText=cell_text2, colWidths=[.8], 
                       bbox=[0.1, 0.76, 1.2, .1], colLabels=["LassoCV"],  rowColours=['#dce9fa', '#bfcad9', '#dce9fa', '#bfcad9', '#dce9fa'], 
                       colColours=['#bfcad9'], cellColours=[['#dce9fa'], ['#bfcad9'], ['#dce9fa'], ['#bfcad9']]
)

df_t_table.auto_set_font_size(False)
df_t_table.set_fontsize(10)
df_t_table.scale(1.1, 1.2)


################################################## Formatting for LassoCV Coef_ table #######################################################
ax = axes[2]

row_labels3 = df_t_coef['Features'].to_list()
cell_text3 = [[str(val)] for val in df_t_coef['Coefficients']]

row_colors = [None] * len(row_labels3)
colors = ['#dce9fa', '#bfcad9']

for i, feature in enumerate(row_labels3):
    row_colors[i] = colors[i % len(colors)]

cell_colors = [[val] for val in row_colors]

df_t_coef_table = ax.table(rowLabels=row_labels3, cellText=cell_text3, colWidths=[.8], 
                       loc='center', colLabels=["Coefficients"],  rowColours=row_colors, colColours=['#bfcad9','#bfcad9'], cellColours=cell_colors
)

df_t_coef_table.auto_set_font_size(False)
df_t_coef_table.set_fontsize(10)
df_t_coef_table.scale(1.1, 1.1)



################################################## Formatting for LassoCv Coef_ plot #######################################################
ax = axes[1]

df_t_coef = df_t_coef.iloc[::-1]
x_val = [val for val in df_t_coef['Coefficients']]
y_val = df_t_coef['Features'].to_list()

ax.barh(width=x_val ,y=y_val, height=.8, color="#bfcad9")
ax.axvline(x=0, color='#000000', linestyle='--', linewidth=1)

ax.set_xlabel('Coefficients')
ax.set_ylabel('Features')


################################################## Formatting for LazyRegressor Plot #######################################################
ax = axes[0]


# df_t_coef = df_t_coef.iloc[::-1]
reg_models.drop(index=['Lars'], inplace=True)

x_val_2 = [val for val in reg_models['Adjusted R-Squared']]
y_val_2 = reg_models.index

# display(reg_models)

ax.barh(width=x_val_2, y=y_val_2, height=.8, color="#bfcad9")
ax.axvline(x=0, color='#000000', linestyle='--', linewidth=1)

ax.set_xlabel('Adjusted R-Squared')
ax.set_ylabel('Model')

display(reg_models)

plt.tight_layout()
plt.show()
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

ax.barh(width=x_val_2, y=y_val_2, height=.8, color="#bfcad9")
ax.axvline(x=0, color='#000000', linestyle='--', linewidth=1)

ax.set_xlabel('Adjusted R-Squared')
ax.set_ylabel('Model')


plt.tight_layout()
plt.show()
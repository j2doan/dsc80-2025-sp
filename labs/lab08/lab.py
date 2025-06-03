# lab.py


import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from pathlib import Path
from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer
from itertools import combinations
from sklearn.metrics import mean_squared_error



import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def best_transformation():
    return 1


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def create_ordinal(df):
    diamond_dct = {'Fair': 0, 
        'Good': 1,
        'Very Good': 2,
        'Premium': 3, 
        'Ideal': 4,
        'J': 0,
        'I': 1,
        'H': 2,
        'G': 3,
        'F': 4,
        'E': 5,
        'D': 6,
        'I1': 0,
        'SI2': 1,
        'SI1': 2,
        'VS2': 3,
        'VS1': 4,
        'VVS1': 5,
        'VVS2': 6,
        'IF': 7
        }

    def helper(dia, col_name):

        dia[f"ordinal_{col_name}"] = dia[col_name].apply(lambda x: diamond_dct[x])
        dia.drop(col_name, axis=1, inplace=True)

    new_diamonds = df.copy()

    for col in df.columns:
        if df[col].dtype == 'object':
            helper(new_diamonds, col)
        else:
            new_diamonds.drop(col, axis=1, inplace=True)

    return new_diamonds


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def create_one_hot(df):
    def one_hot_encode_column(col_name):
        col = df[col_name]
        unique_vals = col.unique()
        one_hot_df = pd.DataFrame()

        # this is NOT looping over rows
        for val in unique_vals:
            one_hot_col_name = f"one_hot_{col_name}_{val}"
            one_hot_df[one_hot_col_name] = (col == val).astype(int)

        return one_hot_df

    one_hot_encoded = []

    for col in df.columns:
        if df[col].dtype == 'object':
            one_hot_encoded.append(one_hot_encode_column(col))

    return pd.concat(one_hot_encoded, axis=1)


def create_proportions(df):
    encoded_probabilities = pd.DataFrame()

    for col in df.columns:
        if df[col].dtype == 'object':
            counts = df[col].value_counts(normalize=True) # series
            encoded_probabilities[f"proportion_{col}"] = df[col].map(counts)

    return encoded_probabilities


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def create_quadratics(df):

    quan_df = df.select_dtypes(include='number')

    quan_df = quan_df.drop('price', axis=1)

    columns = list(combinations(quan_df.columns, 2))

    output = pd.DataFrame()

    for col in columns:
        i, j = col[0], col[1]
        output[f"{i} * {j}"] = quan_df[i] * quan_df[j]

    return output


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------



def comparing_performance():
    # create a model per variable => (variable, R^2, RMSE) table
    """
    def model(X, y):

        lr = LinearRegression()
        lr.fit(X, y)  # X is a DataFrame of training data; y is a Series of prices
        r2 = lr.score(X, y)  # R-squared
        y_pred = lr.predict(X) # predicted prices
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        return r2, rmse
    """
    return [0.8493305264354858, 1548.5331930613174, 'x', 'carat * x', 'ordinal_color', 1434.8400089047336]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    # Question 6.1
    def transform_carat(self, data):
        binarizer = Binarizer(threshold=1.0)
        return binarizer.fit_transform(data[['carat']])
    
    # Question 6.2
    def transform_to_quantile(self, data):
        qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
        qt.fit(self.data[['carat']])
        return qt.transform(data[['carat']])
    
    # Question 6.3
    def transform_to_depth_pct(self, data):
        def depth(X):
            x, y, z = X['x'], X['y'], X['z']
            with np.errstate(divide='ignore', invalid='ignore'):
                depth_pct = 100 * (2 * z) / (x + y)
                depth_pct[np.isinf(depth_pct)] = np.nan
            return depth_pct.to_numpy()

        transformer = FunctionTransformer(depth)
        return transformer.transform(data)

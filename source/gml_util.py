"""
A utility library for Google Machine Learning Crash Course project.
"""

import numpy as np

# Import movie data file
movie_data_path = 'data/movie_metadata.csv'

def print_data_shape(data, des='Data'):
    print('{0} shape: {1}'.format(des, data.shape))


def __inf_to_nan(series):
    return series.replace([np.inf, -np.inf], np.nan)

def __drop_na_rows(df, columns):
    """
    remove those rows in df which has nan in one of the columns
    """
    return df.dropna(axis=0, how='any', subset=columns)

def clean(data):
    data['title_year'] = __inf_to_nan(data['title_year'])
    data = __drop_na_rows(data, ['title_year'])
    return data

def get_us_only(data):
    return data[data.country == 'USA']

def get_does_have_gross(data):
    return data[data.gross > 0]

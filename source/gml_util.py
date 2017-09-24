"""
A utility library for Google Machine Learning Crash Course project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from tensorflow.contrib.learn.python.learn import learn_io

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

def filtered_available_features(input_data):
    """Prepares input features from data set.
    Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
    """
    selected_features = input_data[['duration',
       'director_facebook_likes', 'actor_3_facebook_likes', 
       'actor_1_facebook_likes', 
       'cast_total_facebook_likes',
       'facenumber_in_poster',
       'title_year', 'actor_2_facebook_likes']]
    processed_features = selected_features.copy()
    
    # Create a synthetic feature?
    
    return processed_features


def get_targets(input_data):
    """Prepares target features (i.e., labels)
    Returns:
    A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    
    # Scale the target to be in units of millions of dollars.
    output_targets["adjusted_profit"] = (input_data["adjusted_profit"] / 1000000.0)
    return output_targets

def __mse(arr1, arr2):
    if len(arr1) != len(arr2):
        return None
    length = len(arr1)
    return math.sqrt(np.sum(np.power(np.array(arr1).reshape(length) - np.array(arr2).reshape(length), 2)))

def train_model(
    target,
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    """Trains a linear regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    ...

    Returns:
    A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    feature_columns = set([tf.contrib.layers.real_valued_column(my_feature) for my_feature in training_examples])
    linear_regressor = tf.contrib.learn.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
      gradient_clip_norm=5.0
    )

    # Create input functions
    training_input_fn = learn_io.pandas_input_fn(
      x=training_examples, y=training_targets[target],
      num_epochs=None, batch_size=batch_size)
    predict_training_input_fn = learn_io.pandas_input_fn(
      x=training_examples, y=training_targets[target],
      num_epochs=1, shuffle=False)
    predict_validation_input_fn = learn_io.pandas_input_fn(
      x=validation_examples, y=validation_targets[target],
      num_epochs=1, shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.fit(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # Take a break and compute predictions.
        training_predictions = list(linear_regressor.predict(input_fn=predict_training_input_fn))
        validation_predictions = list(linear_regressor.predict(input_fn=predict_validation_input_fn))
        # Compute training and validation loss.
        training_root_mean_squared_error = __mse(training_predictions, training_targets)
        validation_root_mean_squared_error = __mse(validation_predictions, validation_targets)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")


    # Output a graph of loss metrics over periods.
    plt.xlabel("Periods")
    plt.ylabel("RMSE")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    return linear_regressor
  
def categorize(arr, num_category=15, use_normal_dist=False):
    counts = []
    low_highs = []
    min_val = np.min(arr)
    max_val = np.max(arr)
    mean_val = np.mean(arr)
    if use_normal_dist:
        interval = np.std(arr)/(num_category/2)
    else:
        interval = (max_val - min_val) / num_category
    for i in range(num_category):
        low = min_val + i * interval
        if i is num_category - 1:
            high = max_val
        else:
            high = low + interval
        count = np.count_nonzero(np.logical_and(arr >= low, arr < high))
        counts.append(count)
        low_highs.append((low, high))
    return low_highs, counts

def plot_counts(x_labels, counts):
    plt.bar(np.arange(len(x_labels)), counts)
    plt.figure(figsize=(30, 20))
    plt.show()
    

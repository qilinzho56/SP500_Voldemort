from sp500.time_series.time_series_preprocessing import test_train_prep
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import reciprocal
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
import joblib
from pathlib import Path
import os
tf.random.set_seed(42)
np.random.seed(42)

ANN_PARAMS = {
    "model__n_hidden": [0, 1, 2, 3],
    "model__n_neurons": np.arange(1, 100).tolist(),
    "model__learning_rate": reciprocal(3e-4, 1e-2).rvs(1000).tolist(),
}

RNF_PARAMS = {
    "max_depth": np.arange(10, 110, 10),
    "min_samples_leaf": np.arange(1, 50).tolist(),
    "min_samples_split": np.arange(2, 120, 4).tolist(),
    "n_estimators": np.arange(200, 2000, 200).tolist(),
}

LSTM_PARAMS = {
    "model__n_hidden": [0, 1, 2, 3],
    "model__n_neurons": np.arange(1, 100).tolist(),
    "model__learning_rate": reciprocal(3e-4, 1e-2).rvs(1000).tolist(),
}


# Train three machine learning models
# Random Forest
# ANN
# LSTM
def rnf_model(
    n_estimators=200,
    min_samples_split=100,
    min_samples_leaf=2,
    max_depth=10,
):
    """
    Creates and returns a RandomForestClassifier instance

    Parameters
    ----------
    n_estimators (int): the number of trees to build within a Random Forest before aggregating the predictions
    min_samples_split (int): sets the minimum number of samples that must be present in order for a split to occur
    min_samples_leaf (int): determines minimum size of the end node of each decision tree
    max_depth (int): max depth of the tree

    Returns
    ----------
        A RandomForestClassifier with predefined hyperparameters
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=42,
    )


# Hyperparameter Tuning for ANN
def ann_model_builder(input_shape):
    """
    Returns a function that builds and compiles an ANN Keras Sequential model for binary classification,
    based on the given input shape and other hyperparameters. Use wrapper function, due to
    scikeras.wrappers functionality.

    Parameters
    ----------
    input_shape: an integer recording the number of features of the input layer

    Returns
    -------
    model_fn: a function that returns a compiled Keras model, when called
    """

    def model_fn(n_hidden=1, n_neurons=30, learning_rate=0.01, activation="relu"):
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(input_shape,)))
        model.add(keras.layers.Flatten())

        for _ in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation=activation))

        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )
        return model

    return model_fn


def create_sequences(all_data, past_days=100, future_day=1):
    """
    Create sequence for LSTM model, which requires 3D input,
    (samples, time steps, features)

    Parameters
    ----------
    all_data: a dataframe containing full observations of features and class labels
    past_days (int): number of past observations on daily frequency
    future_day (int): number of observations for prediction on daily freqeuncy

    Returns
    -------
    two numpy arrays: one is the sequence of past observations (3D), the other is the sequence of labels (2D)
    """
    X, y = [], []

    for i in range(past_days, len(all_data) - future_day + 1):
        X.append(all_data.iloc[i - past_days : i, :-1])
        y.append(all_data.iloc[i + future_day - 1, -1])

    return np.array(X), np.array(y)


def lstm_test_train(all_data):
    """
    Create test and training data for LSTM model

    Parameters
    ----------
    all_data: a dataframe containing full observations of features and class labels

    Returns
    -------
    a 4-tuple: (X_train, X_test, y_train, y_test)
    """
    X, y = create_sequences(all_data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def lstm_builder(num_features, past_days=100):
    """
    Returns a function that builds and compiles an LSTM Keras Sequential model for binary classification,
    based on the given input shape and other hyperparameters. Use wrapper function, due to
    scikeras.wrappers functionality.

    Parameters
    ----------
    num_features (int): number of features for prediction
    past_days (int): number of past observations on daily frequency

    Returns
    -------
    model_fn: a function that returns a compiled Keras model, when called
    """

    def model_fn(n_hidden=1, n_neurons=50, learning_rate=0.01):
        model = keras.models.Sequential()
        model.add(
            keras.layers.LSTM(
                n_neurons,
                return_sequences=(n_hidden > 0),
                input_shape=[past_days, num_features],
            )
        )

        for i in range(1, n_hidden):
            model.add(keras.layers.LSTM(n_neurons, return_sequences=(i < n_hidden - 1)))

        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )
        return model

    return model_fn


def rnd_best_params(data, model_type, param_dist):
    """
    Performs randomized search with cross validation to find the best hyperparameters 
    for the specified model type; returns best parameters for future reference
    and also saves the best model as local files

    Parameters
    ----------
        data: a 5-tuple containing training and test datasets: (all_data, X_train, X_test, y_train, y_test)
        model_type: a string of model name
        param_dist: a dictionary containing distribution of parameters to sample during the randomized search

    Returns
    ----------
        best_params: a dictionary containing the best hyperparameters found during the randomized search
    """
    all_data, X_train, _, y_train, _ = data
    model_dir = Path(__file__).parent

    if model_type == "rnf":
        rnf_clf = rnf_model()
        rnf_model_path = model_dir / "best_rnf_model.joblib"
        rnd_search_cv = RandomizedSearchCV(rnf_clf, param_dist, n_iter=10, cv=3, random_state=42)
        joblib.dump(rnd_search_cv.fit(X_train, y_train).best_estimator_, rnf_model_path)

    if model_type == "ann":
        X_train = data[1]
        input_shape = X_train.shape[1]
        model_fn = ann_model_builder(input_shape)

    if model_type == "lstm":
        X_train, X_test, y_train, y_test = lstm_test_train(all_data)
        model_fn = lstm_builder(X_train.shape[2])

    if model_type in ["ann", "lstm"]:
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            str(model_dir / f"best_{model_type}_model.h5"), save_best_only=True
        )
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True
        )

        classifier = KerasClassifier(
            build_fn=model_fn,
            epochs=20,
            validation_split=0.2,
            callbacks=[checkpoint_cb, early_stopping_cb],
        )

        rnd_search_cv = RandomizedSearchCV(classifier, param_dist, n_iter=10, cv=3, random_state=42)
        rnd_search_cv.fit(X_train, y_train)

    return rnd_search_cv.best_params_


def predict_with_best_model(
    data,
    best_model,
    threshold=0.5,
    model_type="rnf",
    new_test_data=None,
    backtest=False,
):
    """
    Merger ticker_df with dataframes in macro_indicators, with best model
    found in the randomized search

    Parameters
    ----------
    data: a 5-tuple containing training and test datasets: (all_data, X_train, X_test, y_train, y_test)
    threshold: a float which determines whether the predicted class is 1 or 0
    model_type: a string of classifier name
    new_test_data: a 2-tuple containing test arrays for LSTM
    backtest (boolean): whether or not utilize backtesting

    Returns
    -------
    preds_summary: a dataframe containing actual labels, predicted labels, with two associated class
    probabilities
    """
    # Condition Check: LSTM and backtesting
    # LSTM is purely based on numpy arrays, but Random Forest and Artifical Neural Network
    # involves dataset and time indices for direct visualization
    if model_type == "lstm" and backtest:
        X_test, y_test = new_test_data
    elif model_type == "lstm" and not backtest:
        all_data, _, _, _, _ = data
        _, X_test, _, y_test = lstm_test_train(all_data)
    else:
        _, X_train, X_test, y_train, y_test = data

    # Seperate prediction cases based on differed functionality of each model
    if model_type == "rnf":
        best_model.fit(X_train, y_train)
        test_probs = best_model.predict_proba(X_test)
        preds = (test_probs[:, 1] >= threshold).astype(int)
        neg_probs = pd.Series(test_probs[:, 0], index=y_test.index)
        pos_probs = pd.Series(test_probs[:, 1], index=y_test.index)

    if model_type in ["ann", "lstm"]:
        test_probs = best_model.predict(X_test).squeeze()
        preds = (test_probs >= threshold).astype(int)
        if model_type == "ann":
            neg_probs = pd.Series(1 - test_probs, index=y_test.index)
            pos_probs = pd.Series(test_probs, index=y_test.index)
        else:
            neg_probs = pd.Series(1 - test_probs)
            pos_probs = pd.Series(test_probs)

    preds_summary = pd.DataFrame(
        {
            "Actual": y_test,
            "Predictions": preds,
            "Negative_Probability": neg_probs,
            "Positive_Probability": pos_probs,
        }
    )

    return preds_summary


def backtest(data, best_model, model_type, start=1008, step=252):
    """
    Backtesting the model with five-year as a cycle, i.e.,
    take the first 4 years of data and use it to predict
    values for the 5th year

    Assuming 252 days in a trading year

    Parameters
    ----------
    data: a 5-tuple after test_train splitting
    model_type: a string of classifier name
    start: an integer of years for training
    step: an integer of days in a trading year to move to the year to be predicted

    Returns
    -------
    a dataframe containing actual labels, predicted labels, with two associated class
    probabilities, under backtesting system
    """
    all_data, _, _, _, _ = data
    all_predictions = []

    if model_type == "lstm":
        X_train, X_test, y_train, y_test = lstm_test_train(all_data)
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        for i in range(start, X.shape[0], step):
            if i + step > X.shape[0]:
                break
            X_train = X[0:i, :, :]
            X_test = X[i : i + step, :, :]
            y_train = y[0:i]
            y_test = y[i + step]
            new_data = (X_test, y_test)
            predictions = predict_with_best_model(
                data=None,
                best_model=best_model,
                model_type=model_type,
                new_test_data=new_data,
                backtest=True,
            )
            all_predictions.append(predictions)

    else:
        for i in range(start, all_data.shape[0], step):
            X_train = all_data.iloc[0:i, :-1]
            X_test = all_data.iloc[i : i + step, :-1]
            y_train = all_data.iloc[0:i, -1]
            y_test = all_data.iloc[i : i + step, -1]

            new_data = (None, X_train, X_test, y_train, y_test)
            predictions = predict_with_best_model(
                data=new_data, best_model=best_model, model_type=model_type
            )
            all_predictions.append(predictions)

    return pd.concat(all_predictions)


def valuation_metric(y_test, y_preds):
    """
    Design a valuation metric report for predictions

    Parameters
    ----------
    y_test: a pd.Series containing actual class labels
    y_preds: a pd.Series containing predicted class labels

    Returns
    -------
    a statistics dataframe: including accuracy, precision, original_positive_percentage, recall, f1score
    """
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1score = f1_score(y_test, y_preds)
    original_pos_percent = (y_test == 1).mean()

    stats_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Original Positive Percentage', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, original_pos_percent, recall, f1score]
    })

    # Back testing outputs optional
    """
    print(f"Accuracy = {accuracy}")
    print(f"Precision = {precision}")
    print(f"Test Positive Percentage = {original_pos_percent}")
    print(f"Recall = {recall}")
    print(f"F1 Score = {f1score}")
    """
    return stats_df


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    data = test_train_prep("AAPL")
    keras.backend.clear_session()

    best_params =  rnd_best_params(data, model_type="ann", param_dist=ANN_PARAMS)
    best_model = keras.models.load_model("best_ann_model.h5")
    # best_model = joblib.load("best_rnf_model.joblib")
    general_preds = predict_with_best_model(data, best_model, model_type="ann")
    valuation_metric(general_preds["Actual"], general_preds["Predictions"])

    backtest_preds = backtest(data, best_model, "ann")
    valuation_metric(backtest_preds["Actual"], backtest_preds["Predictions"])

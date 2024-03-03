from sp500.time_series.time_series import test_train_prep
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import reciprocal
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier

tf.random.set_seed(42)

ANN_PARAMS = {
    "model__n_hidden": [0, 1, 2, 3],
    "model__n_neurons": np.arange(1, 100).tolist(),
    "model__learning_rate": reciprocal(3e-4, 1e-2).rvs(1000).tolist(),
}

RNF_PARAMS = {
    "max_depth": np.arange(10, 110, 10),
    "max_features": ["auto", "sqrt"],
    "min_samples_leaf": np.arange(1, 5).tolist(),
    "min_samples_split": np.arange(2, 12, 4).tolist(),
    "n_estimators": np.arange(200, 2000, 200).tolist(),
}


# Train three machine learning models
# Random Forest
# ANN
# RNN
def rnf_model():
    """
    Creates and returns a RandomForestClassifier instance

    Returns:
        model: A RandomForestClassifier with predefined hyperparameters
    """
    model = RandomForestClassifier(
        n_estimators=200, min_samples_split=100, random_state=42
    )
    return model


# Hyperparameter Tuning for ANN
def ann_model_builder(input_shape):
    """
    Returns a function that builds and compiles a Keras Sequential model for binary classification,
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
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return model

    return model_fn


def rnd_best_params(data, model_type, param_dist):
    """
    Performs randomized search with cross validation to find the best hyperparameters for the specified model type

    Parameters
    ----------
        data: a 5-tuple containing training and test datasets: (all_data, X_train, X_test, y_train, y_test)
        model_type: a string of model name 
        param_dist: a dictionary containing distribution of parameters to sample during the randomized search

    Returns
    ----------
        best_params: a dictionary containing the best hyperparameters found during the randomized search
    """
    _, X_train, X_test, y_train, y_test = data

    if model_type == "rnf":
        classifier = rnf_model()

    if model_type == "ann":
        X_train = data[1]
        input_shape = X_train.shape[1]
        model_fn = ann_model_builder(input_shape)

        checkpoint_cb =  tf.keras.callbacks.ModelCheckpoint(
            "best_" + model_type + "_model.h5", save_best_only=True
        )
        early_stopping_cb =  tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True
        )

        classifier = KerasClassifier(
            build_fn=model_fn,
            epochs=20,
            validation_split=0.2,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )

    rnd_search_cv = RandomizedSearchCV(classifier, param_dist, n_iter=10, cv=3)
    rnd_search_cv.fit(X_train, y_train)

    return rnd_search_cv.best_params_


def predict_with_best_model(data, best_model, threshold=0.5, model_type="rnf"):
    """
    Merger ticker_df with dataframes in macro_indicators, with best model
    found in the randomized search

    Parameters
    ----------
    data: a 5-tuple containing training and test datasets: (all_data, X_train, X_test, y_train, y_test)
    threshold: a float which determines whether the predicted class is 1 or 0
    model_type: a string of classifier name

    Returns
    -------
    preds_summary: a dataframe containing actual labels, predicted labels, with two associated class
    probabilities
    """
    _, X_train, X_test, y_train, y_test = data

    if model_type == "rnf":
        test_probs = best_model.predict_proba(X_test)
        preds = (test_probs[:, 1] >= threshold).astype(int)
        neg_probs = pd.Series(test_probs[:, 0], index=y_test.index)
        pos_probs = pd.Series(test_probs[:, 1], index=y_test.index)

    if model_type == "ann":
        test_probs = best_model.predict(X_test).flatten()
        preds = (test_probs >= threshold).astype(int)
        neg_probs = pd.Series(1 - test_probs, index=y_test.index)
        pos_probs = pd.Series(test_probs, index=y_test.index)

    preds_summary = pd.DataFrame(
        {
            "Actual": y_test,
            "Predictions": preds,
            "Negative_Probability": neg_probs,
            "Positive_Probability": pos_probs,
        },
        index=y_test.index,
    )

    return preds_summary


def backtest(data, best_model, model_type, start=1260, step=252):
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

    for i in range(start, all_data.shape[0], step):
        X_train = all_data.iloc[0:i, :-1].copy()
        X_test = all_data.iloc[i : i + step, :-1].copy()
        y_train = all_data.iloc[0:i, -1].copy()
        y_test = all_data.iloc[i : i + step, -1].copy()

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
    a 5-tuple: (accuracy, precision, original_positive_percentage, recall, f1score)
    """
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1score = f1_score(y_test, y_preds)
    original_pos_percent = (y_test == 1).mean()

    print(f"Accuracy = {accuracy}")
    print(f"Precision = {precision}")
    print(f"Test Positive Percentage = {original_pos_percent}")
    print(f"Recall = {recall}")
    print(f"F1 Score = {f1score}")
    return (accuracy, precision, original_pos_percent, recall, f1score)


if __name__ == "__main__":

    keras.backend.clear_session()
    data = test_train_prep("AAPL")
    X_train = data[1]
    best_params = rnd_best_params(data, "ann", ANN_PARAMS)
    best_model = keras.models.load_model("best_ann_model.h5")
    general_preds = predict_with_best_model(data, best_model, model_type="ann")
    valuation_metric(general_preds["Actual"], general_preds["Predictions"])

    backtest_preds = backtest(data, best_model, "ann")
    valuation_metric(backtest_preds["Actual"], backtest_preds["Predictions"])
from sp500.time_series.time_series import test_train_prep
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)

def rnf_model(n_estimators=200, min_samples_split=100, random_state=42):
    model = RandomForestClassifier(n_estimators, min_samples_split, random_state)
    return model


def ann_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")  
    ])
    
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy", 
                           tf.keras.metrics.Precision(), 
                           tf.keras.metrics.Recall()])
    
    return model


def train_predict(company, threshold=0.5, model_type="rnf"):
    X_train, X_test, y_train, y_test = test_train_prep(company)
    if model_type == "rnf":
        rnf_clf = rnf_model.fit(X_train, y_train)
        test_probs = rnf_clf.predict_proba(X_test)
        preds = (test_probs[:, 1] >= threshold).astype(int)
    elif model_type == "ann":
        ann_clf = ann_model.fit(X_train, y_train, epochs=30)
        test_probs = ann_clf.predict(X_test)
        preds = (test_probs >= threshold).astype(int)

    preds_series = pd.Series(preds, index=y_test.index, name="Predictions")
    neg_probs = pd.Series(test_probs[:, 0], name="Negative_Probability")
    pos_probs = pd.Series(test_probs[:, 1], name="Positive_Probability")

    preds_summary = pd.concat([y_test, preds_series, neg_probs, pos_probs], axis=1)
    return preds_summary 


def valuation_metric(y_test, y_preds):
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1score = f1_score(y_test, y_preds)
    original_pos_percent = (y_test == 1).mean() * 100

    print(f"Accuracy = {accuracy}")
    print(f"Precision = {precision}")
    print(f"Test Positive Percentage = {original_pos_percent}")
    print(f"Recall = {recall}")
    print(f"F1 Score = {f1score}")
    return (accuracy, precision, original_pos_percent, recall, f1score)


if __name__ == "__main__":
    rnf_preds = train_predict(company="AAPL")
    valuation_metric( rnf_preds[0], rnf_preds["Predictions"])

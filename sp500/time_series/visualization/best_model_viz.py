from sp500.time_series.price_model import rnd_best_params, predict_with_best_model, valuation_metric, backtest
from sp500.time_series.time_series_preprocessing import test_train_prep
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import numpy as np
import joblib
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

DIR = Path(__file__).parents[1]
ANN = keras.models.load_model(DIR / "best_ann_model.h5")
LSTM = keras.models.load_model(DIR / "best_lstm_model.h5")
RNF = joblib.load(DIR / "best_rnf_model.joblib")
MODELS = {"rnf": RNF, "ann": ANN, "lstm": LSTM}

# Visualization for the saved best trained model 
def plot_predictions_summary(preds_df, company, model_type):
    """
    Plots the actual vs. predicted labels and probability distributions for class labels

    Parameters
    ----------
    - preds_df: a dataFrame containing actual labels, predicted labels,
      and the associated class probabilities
    - company (string): company name
    - model_type (string): model name

    Returns
    -------
    fig1, fig2: figure objects for the actual vs. predicted plot and the probability distributions plot
    """
    conf_matrix = confusion_matrix(preds_df["Actual"], preds_df["Predictions"])
    # Referenced https://medium.com/mlearning-ai/heatmap-for-correlation-matrix-confusion-matrix-extra-tips-on-machine-learning-b0377cee31c2
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues', ax=ax1, cbar=False)
    ax1.set_title(f"Confusion Matrix - {company} ({model_type})")
    ax1.set_xlabel("Predicted Labels")
    ax1.set_ylabel("Actual Labels")
    ax1.set_xticklabels(["Down", "Up"])
    ax1.set_yticklabels(["Down", "Up"])

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(preds_df["Positive_Probability"], bins=40, alpha=0.6, color="red", label="Positive Probability", density=True)
    ax2.hist(preds_df["Negative_Probability"], bins=40, alpha=0.6, color="gray", label="Negative Probability", density=True)
    ax2.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    ax2.set_title(f"Probability Distributions - {company} ({model_type})")
    ax2.set_xlabel("Probability")
    ax2.set_ylabel("Density")
    ax2.legend()

    return fig1, fig2

def model_summary_figs(data, company, models=MODELS):
    """
    Generates and returns prediction summary plots and accuracy metrics for a given company 
    using best models

    Parameters
    ----------
    data: the five-tuple containing all_data, X_train, y_train, X_test, y_test 
    company: a string of company name
    models: a dictionary mapping model names to their trained models
    
    Returns:
    ----------
    preds_summary_plots (dict): a dctionary mapping model types to their prediction summary plots
    accuracy_tables (dict): a dictionary mapping model types to their accuracy metrics tables
    """
    preds_summary_plots = {}
    accuracy_tables = {}

    for model_type, best_model in MODELS.items():
        preds_df = predict_with_best_model(data=data, best_model=best_model, model_type=model_type)
        preds_summary_plots[model_type] = plot_predictions_summary(preds_df, company, model_type)
        accuracy_tables[model_type] = valuation_metric(preds_df["Actual"],  preds_df["Predictions"])
        
    return preds_summary_plots, accuracy_tables

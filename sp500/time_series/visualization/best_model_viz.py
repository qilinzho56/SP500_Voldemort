from sp500.time_series.price_model import rnd_best_params, predict_with_best_model, valuation_metric, backtest
from sp500.time_series.time_series_preprocessing import test_train_prep
from tensorflow import keras
import joblib
from pathlib import Path
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

    fig1, ax1 = plt.subplots(figsize=(300, 300))
    ax1.plot(preds_df["Actual"].reset_index(drop=True), color="blue", label="Actual")
    ax1.scatter(range(len(preds_df)), preds_df["Predictions"], color="red", label="Predicted", alpha=0.5, edgecolor="none")
    ax1.set_title(f"Actual vs. Predicted Labels - {company} ({model_type})")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Class")
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(300, 300))
    ax2.hist(preds_df["Positive_Probability"], bins=30, alpha=0.6, color="red", label="Positive Probability", density=True)
    ax2.hist(preds_df["Negative_Probability"], bins=30, alpha=0.6, color="gray", label="Negative Probability", density=True)
    ax2.set_title(f"Probability Distributions - {company} ({model_type})")
    ax2.set_xlabel("Probability")
    ax2.set_ylabel("Density")
    ax2.legend()

    return fig1, fig2

def model_summary_figs(company):
    """
    Generates and returns prediction summary plots and accuracy metrics for a given company 
    using best models

    Parameters
    ----------
    company (str): company name

    Returns:
    ----------
    preds_summary_plots (dict): a dctionary mapping model types to their prediction summary plots
    accuracy_tables (dict): a dictionary mapping model types to their accuracy metrics tables
    """
    preds_summary_plots = {}
    accuracy_tables = {}
    data = test_train_prep(company)

    for model_type, best_model in MODELS.items():
        preds_df = predict_with_best_model(data=data, best_model=best_model, model_type=model_type)
        preds_summary_plots[model_type] = plot_predictions_summary(preds_df, company, model_type)
        accuracy_tables[model_type] = valuation_metric(preds_df["Actual"],  preds_df["Predictions"])
        
    return preds_summary_plots, accuracy_tables

if __name__ == "__main__":
    model_summary_figs("AAPL")
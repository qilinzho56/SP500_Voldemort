from sp500.time_series.time_series_preprocessing import (
    preprocess_macro_data,
    fetch_macro_indicators,
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_macro_indicators(macro_indicators):
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    lines, labels = [], []

    for _, df in macro_indicators.items():
        df.index = pd.to_datetime(df.index)
        df = df.where(df != "NA", np.nan).dropna()
        column_name = df.columns[0]

        if column_name == "Foreign Exchange Rates":
            line = ax2.plot(
                df.index,
                df[column_name],
                label=column_name + " (Right Y-Axis)",
                marker="",
                linestyle="-",
                linewidth=1.0,
                color="red",
            )
            lines += line

        else:
            line = ax1.plot(
                df.index,
                df[column_name],
                label=column_name,
                marker="",
                linestyle="-",
                linewidth=1.0,
            )
            lines += line
    labels += [l.get_label() for l in lines]

    plt.title("Macroeconomic Indicators Over Time")
    plt.xlabel("Date")
    plt.ylabel("Rate")
    plt.legend(lines, labels)

    filename = "Macro Indicators.png"
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    macro_indicators = fetch_macro_indicators()
    preprocess_macro_data(macro_indicators)
    plot_macro_indicators(macro_indicators)

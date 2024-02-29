# 📈 S&P 500 Stock Movement Prediction

## 🌟 Overview
This repository hosts the code 🧑‍💻 and documentation 📄 for a machine learning project aimed at predicting stock market movements. We use various data sources 📊, factor analysis 🔍, and machine learning techniques 🤖, including sentiment analysis and time-series forecasting.

## 📚 Table of Contents
- [Literature Review](#literature-review)
- [Package Exploration](#package-exploration)
- [Data Collection, Analysis, and Evaluation](#data-collection-analysis-and-evaluation)
- [Factor Analysis](#factor-analysis)
- [Development Process](#development-process)

![Project Scheme](Scheme.png)

## 📖 Literature Review
A thorough investigation of existing research on stock movement prediction methodologies.

## 🔎 Package Exploration
Examination of various Python libraries 🐍 for data analysis and machine learning.

## 🗂️ Data Collection, Analysis, and Evaluation
Details the data sources and analytical methods used in the project.

### 🛠️ Data Sources
- **Stock Data**: Collected through Yahoo Finance API 📈.
- **Economic Indicators**: Fetched from DBnomics API for FED and OECD data 🌐.

## 📈 Factor Analysis
Explores the relationship between expected returns and systematic risk.
- **Sentiment Analysis**: Implemented to understand market sentiment's impact on stock movements 💬.
- **Time-Series Data**: Analyzed to capture patterns and trends over time ⏳.

## 💻 Development Process
Documentation of the development steps, including data collection and GUI creation 🖥️.

## ❓ What to Do Next?
Outlines the forthcoming steps in the project's lifecycle

## How to Use?
### 1.0 Download

```shell
pip install --user -U nltk
```

### 1.1 File Structure
```python
sp500/
    compile/
        __init__.py
        cleanup.py
        pos.py
    headlines/
        __init__.py
        app.py
        displayer.py
        scraper.py
    sa/
        data/
        __init__.py
        analyzer.py
        sa.py
        test.py
        train_classifier.py
    time_series/
        price_model.py
        time_series.py
    visualization/
        __init__.py
        company_profile.py
        create_word_clooud.py
        datatypes.py
poetry.lock
pyproject.toml
README.md
Scheme.png


```
### 1.2 Instruction
1. 
2. 


## Reference
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
- Loughran, T. and McDonald, B. (2011), ``When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.'' The Journal of Finance, 66: 35-65.


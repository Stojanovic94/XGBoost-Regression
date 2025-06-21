# XGBoost Regression on Boston Housing Dataset

This repository contains a practical example of using an XGBoost Regressor to predict house prices based on the Boston Housing dataset.

## Features

- Data preprocessing with categorical encoding (OneHotEncoder)
- Use of sklearn Pipeline for clean and efficient workflow
- Model training, prediction, and evaluation (Mean Squared Error)
- Feature importance extraction and visualization

## Dataset

The Boston Housing dataset is downloaded directly from Kaggle via `kagglehub`. It contains various numerical and categorical features used to predict median house prices.

## Requirements

```bash
pip install pandas scikit-learn matplotlib seaborn xgboost kagglehub

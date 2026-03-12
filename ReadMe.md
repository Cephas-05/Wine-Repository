Wine Quality Prediction using Random Forest
Project Overview

This project builds a machine learning model to predict wine quality using physicochemical properties of wine. The goal is to analyze feature relationships, handle class imbalance, and improve prediction performance using Random Forest and hyperparameter tuning.

The project explores multiple techniques including:

Exploratory Data Analysis (EDA)

Feature importance analysis

Handling class imbalance using SMOTE

Hyperparameter tuning using Validation Curves and RandomizedSearchCV

Comparing balanced vs unbalanced training strategies

Dataset

The dataset used in this project is WineQT.csv, which contains various chemical attributes of wine samples along with their quality scores.

Features include

Fixed acidity

Volatile acidity

Citric acid

Residual sugar

Chlorides

Free sulfur dioxide

Total sulfur dioxide

Density

pH

Sulphates

Alcohol

Target Variable

quality — wine quality rating.

Exploratory Data Analysis

The following analyses were performed:

Quality Distribution

A count plot was used to observe the distribution of wine quality values.

Feature Relationship Analysis

Bar plots were used to analyze relationships between features and wine quality.

Key observations:

Volatile acidity has an inverse relationship with wine quality.

Citric acid tends to increase as wine quality increases.

These insights suggest that chemical properties influence wine quality.

Data Preprocessing

The dataset was split into training and testing sets:

80% training

20% testing

train_test_split(x, y, test_size=0.2, random_state=42)
Handling Class Imbalance

Wine quality classes were imbalanced. Two different techniques were explored.

Method 1: SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE generates synthetic samples for minority classes to balance the dataset.

from imblearn.over_sampling import SMOTE

Training data was balanced using:

SMOTE(random_state=42)

However, this method slightly decreased overall model performance.

Method 2: Class Weight Balancing

Instead of generating synthetic data, class weights were adjusted to penalize misclassification of minority classes.

RandomForestClassifier(class_weight="balanced")

This method produced more stable results compared to SMOTE.

Model Selection

A Random Forest Classifier was chosen due to its advantages:

Handles nonlinear relationships

Robust to noise

Performs well on tabular datasets

Provides feature importance metrics

Feature Importance Analysis

Feature importance was extracted from the trained Random Forest model.

This helps identify which chemical properties contribute most to wine quality prediction.

Visualization was done using Seaborn bar plots.

Hyperparameter Tuning

Two approaches were used.

Validation Curves

Validation curves were used to analyze the effect of specific hyperparameters such as:

n_estimators

max_depth

min_samples_split

Cross-validation was performed and mean scores were plotted to observe overfitting and underfitting trends.

Example:

validation_curve(RandomForestClassifier())
RandomizedSearchCV

Instead of manually tuning parameters, RandomizedSearchCV was used to search for optimal hyperparameter combinations.

Example parameter grid:

n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
bootstrap

This method efficiently sampled parameter combinations using cross-validation.

Best Hyperparameters Found

Example output from RandomizedSearchCV:

{
 'n_estimators': 800,
 'min_samples_split': 5,
 'min_samples_leaf': 1,
 'max_features': 'log2',
 'max_depth': 40,
 'bootstrap': True
}
Model Evaluation

The model was evaluated using:

Accuracy

Precision

Recall

F1 Score

Classification Report

accuracy_score()
classification_report()
Results Comparison

Three model setups were compared:

Model	Accuracy
Default Random Forest	Baseline
Random Forest + SMOTE	Slightly lower
Random Forest + Class Weight Balancing	More stable

Although the unbalanced dataset produced slightly higher accuracy, balanced models are preferred for fair classification across classes.

Key Learnings

This project demonstrates several important machine learning concepts:

Class imbalance can significantly affect model performance

SMOTE does not always improve results

Hyperparameter tuning helps explore model behavior

Accuracy alone is not always the best evaluation metric

Feature importance can reveal meaningful relationships in the data

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Imbalanced-learn

Future Improvements

Potential improvements for this project include:

Trying other models such as XGBoost or Gradient Boosting

Performing feature selection

Using Stratified Cross Validation

Deploying the model with Streamlit or FastAPI

Author

Cephas Princely
AI / Machine Learning Engineer
GitHub: https://github.com/Cephas-05

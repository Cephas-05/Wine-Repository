# 🍷 Wine Quality Prediction using Random Forest

## 📌 Project Overview
This project builds a **Machine Learning model to predict wine quality** using physicochemical properties of wine.  
The objective is to analyze the relationship between chemical attributes and wine quality and improve prediction performance using **Random Forest and hyperparameter tuning**.

This project demonstrates several important machine learning concepts including:

- Exploratory Data Analysis (EDA)
- Feature importance analysis
- Handling class imbalance using **SMOTE**
- Hyperparameter tuning using **Validation Curves** and **RandomizedSearchCV**
- Comparing **balanced vs unbalanced training strategies**

---

# 📂 Dataset

The dataset used in this project is **WineQT.csv**, which contains physicochemical properties of wine samples along with their quality ratings.

### Features

The dataset includes the following attributes:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

### Target Variable

`quality` – wine quality score representing the rating of each wine sample.

---

# 📊 Exploratory Data Analysis

EDA was performed to understand the distribution of wine quality and relationships between features.

### Quality Distribution

A **count plot** was used to observe the distribution of wine quality values.

### Feature Relationship Analysis

Bar plots were used to analyze how different features influence wine quality.

### Key Observations

- **Volatile acidity** has an inverse relationship with wine quality.
- **Citric acid** tends to increase as wine quality increases.

These insights suggest that chemical properties influence wine quality.

---

# ⚙️ Data Preprocessing

The dataset was split into training and testing sets.

- **80% Training Data**
- **20% Testing Data**

```
train_test_split(x, y, test_size=0.2, random_state=42)
```

---

# ⚖️ Handling Class Imbalance

Wine quality classes are imbalanced. Two balancing techniques were explored.

---

## Method 1 — SMOTE (Synthetic Minority Oversampling)

SMOTE creates synthetic samples for minority classes.

```
from imblearn.over_sampling import SMOTE
```

```
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
```

However, SMOTE slightly reduced the model's performance for this dataset.

---

## Method 2 — Class Weight Balancing

Instead of generating synthetic samples, **class weights were adjusted** during training.

```
RandomForestClassifier(class_weight="balanced")
```

This approach produced more stable results compared to SMOTE.

---

# 🌳 Model Selection

A **Random Forest Classifier** was chosen because:

- It handles nonlinear relationships
- It performs well on tabular datasets
- It reduces overfitting using ensemble learning
- It provides feature importance metrics

---

# 📈 Feature Importance

Feature importance was extracted from the trained Random Forest model to understand which features contribute most to predictions.

Feature importance was visualized using **Seaborn bar plots**.

---

# 🔧 Hyperparameter Tuning

Two methods were used to tune model hyperparameters.

---

## Validation Curves

Validation curves were used to analyze how different hyperparameters affect model performance.

Hyperparameters explored:

- `n_estimators`
- `max_depth`
- `min_samples_split`

Cross-validation scores were plotted to observe **underfitting and overfitting trends**.

---

## RandomizedSearchCV

Instead of manually tuning parameters, **RandomizedSearchCV** was used to efficiently search for optimal hyperparameters.

Example parameter grid:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features
- bootstrap

RandomizedSearchCV tested multiple combinations using cross-validation.

---

# 🏆 Best Hyperparameters Found

Example output from RandomizedSearchCV:

```
{
 'n_estimators': 800,
 'min_samples_split': 5,
 'min_samples_leaf': 1,
 'max_features': 'log2',
 'max_depth': 40,
 'bootstrap': True
}
```

---

# 📊 Model Evaluation

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

```
accuracy_score()
classification_report()
```

---

# 📉 Results Comparison

Three approaches were compared:

| Model | Result |
|------|------|
| Default Random Forest | Baseline performance |
| Random Forest + SMOTE | Slight decrease in accuracy |
| Random Forest + Class Weight Balancing | More stable results |

Although the unbalanced dataset produced slightly higher accuracy, balanced models help improve predictions for minority classes.

---

# 🧠 Key Learnings

This project highlights several important machine learning insights:

- Class imbalance significantly affects model performance
- SMOTE does not always improve results
- Hyperparameter tuning helps explore model behavior
- Accuracy alone is not sufficient for imbalanced datasets
- Feature importance helps interpret model predictions

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn

---

# 🚀 Future Improvements

Possible improvements for this project include:

- Trying models such as **XGBoost or Gradient Boosting**
- Performing **feature selection**
- Using **Stratified Cross Validation**
- Deploying the model using **Streamlit or FastAPI**

---

# 👨‍💻 Author

**Cephas Princely**

GitHub:  
https://github.com/Cephas-05

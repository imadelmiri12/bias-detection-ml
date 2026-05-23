# Statistical Bias Detection in Machine Learning

Machine Learning and statistical inference project for fairness analysis in binary decision systems using the Adult Income dataset.

---

# Project Overview

This project studies the existence of statistical bias in a binary decision mechanism using Machine Learning and inferential statistics.

A Logistic Regression model is trained on the Adult Income dataset to predict whether an individual's annual income exceeds $50K. The generated decisions are then analyzed statistically to evaluate fairness between demographic groups.

The project combines:

* Machine Learning
* Statistical Inference
* Fairness Analysis
* Bootstrap Methods
* Hypothesis Testing
* Interactive Streamlit Visualization

---

# Objectives

* Build a binary classification model for income prediction
* Analyze statistical bias between demographic groups
* Estimate acceptance probabilities for Male and Female groups
* Construct bootstrap confidence intervals
* Perform hypothesis testing on proportions
* Show that strong predictive performance does not necessarily guarantee fairness

---

# Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Statsmodels
* Matplotlib

---

# Dataset

Dataset used:

* Adult Income Dataset (UCI Machine Learning Repository)

Sensitive attribute analyzed:

* `gender`

Target variable:

* `income > 50K`

---

# Machine Learning Pipeline

## Data Preprocessing

* Missing value handling
* One-Hot Encoding
* Feature selection
* Train/Test split (70% / 30%)

## Models

* Logistic Regression
* Random Forest (comparison)

## Evaluation Metrics

* Accuracy
* ROC Curve
* AUC Score
* Confusion Matrix

---

# Statistical Analysis

The project performs a rigorous statistical analysis of model fairness using:

## Probability Estimation

Estimation of:

* P(Positive Decision | Male)
* P(Positive Decision | Female)

## Bootstrap Analysis

* Non-parametric bootstrap
* Empirical distribution estimation
* 95% confidence intervals

## Hypothesis Testing

* Z-test for proportions
* Statistical significance analysis
* Bias detection between demographic groups

---

# Key Results

## Logistic Regression Performance

* Accuracy ≈ 0.840
* AUC ≈ 0.897

## Statistical Findings

* Significant difference between Male and Female acceptance rates
* Confidence interval does not contain zero
* Strong statistical evidence of indirect bias

Main conclusion:

> Good global model performance does not guarantee fairness between groups.

---

# Streamlit Application

The project includes an interactive Streamlit application with:

* dataset exploration,
* machine learning evaluation,
* ROC visualization,
* bootstrap analysis,
* statistical hypothesis testing,
* fairness comparison dashboards.

---

# Project Structure

```bash id="1k9v8c"
├── app.py
├── adult.csv
├── projet_annoté.ipynb
├── requirements.txt
├── Rapport de Math.pdf
└── README.md
```

---

# Installation

## Clone repository

```bash id="t0d1vq"
git clone https://github.com/your-username/bias-detection-ml.git
cd bias-detection-ml
```

## Install dependencies

```bash id="vgwn2h"
pip install -r requirements.txt
```

## Run Streamlit app

```bash id="e44vyz"
streamlit run app.py
```

---

# Streamlit Features

## Description des données

* dataset exploration
* descriptive statistics
* target distribution

## Machine Learning

* Logistic Regression training
* Random Forest comparison
* ROC/AUC evaluation

## Analyse statistique

* acceptance rate estimation
* bootstrap confidence intervals
* bias visualization

## Test d’hypothèse

* Z-test for proportions
* p-value analysis
* statistical decision

## Comparaison

* ML performance vs fairness analysis

---

# Academic Context

Project developed as part of:

* Master IASD — Intelligence Artificielle et Data Science
* Faculté des Sciences et Techniques de Tanger (FSTT)

---

# Important Insight

This project highlights a critical concept in Responsible AI:

> Removing a sensitive attribute from training data does not necessarily eliminate bias, because indirect correlations may still exist within the dataset.

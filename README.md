# Amazon Employee Access — Kaggle Project

## Overview
This project predicts whether an employee should be granted access to a resource for the **Amazon Access Kaggle competition**. The dataset contains high-cardinality categorical features describing employee roles, departments, resources, and permissions. The goal was to build a full machine-learning workflow in R to compare several classification models and generate accurate Kaggle submissions.

## What the Code Does
- Cleans and preprocesses the data using a tidymodels recipe  
  - Converts all predictors to factors  
  - Handles rare levels + unseen categories  
  - Applies mixed-effects label encoding  
  - Removes zero-variance predictors  
- Trains multiple classification models:  
  - Logistic Regression  
  - Penalized Logistic Regression (glmnet)  
  - Random Forests (ranger)  
  - K-Nearest Neighbors  
  - Naive Bayes  
  - SMOTE workflow for imbalance  
- Tunes models using cross-validation and ROC AUC  
- Generates prediction probabilities (`ACTION = 1`) for the test set  
- Writes Kaggle-ready submission files for each model  



# Titanic Survival Prediction using Artificial Neural Networks

## Project Overview

This project builds a complete machine learning pipeline to predict passenger survival on the RMS Titanic. The primary model is an **Artificial Neural Network (Multi-Layer Perceptron)** , benchmarked against Logistic Regression and Decision Tree classifiers.

This work extends a prior EDA and preprocessing assignment, applying rigorous data preparation, model training, and performance evaluation techniques.

## Key Skills Demonstrated

- **Data Preprocessing:** Handling missing values (median/mode imputation), encoding categorical variables (Label/One-Hot encoding), feature scaling (StandardScaler)
- **Model Development:** Implementing MLP Classifier with ReLU activation, Adam optimizer, and early stopping
- **Evaluation:** Comprehensive model assessment using Accuracy, Precision, Recall, F1-Score, and AUC-ROC
- **Visualization:** Confusion matrices, ROC curves, training loss curves, comparative bar charts
- **Best Practices:** Stratified train-test split (80/20), preventing data leakage, handling class imbalance

## Repository Contents

| File | Description |
|------|-------------|
| `Assignment2_Titanic.ipynb` | Complete Jupyter notebook with all code and outputs |
| `Assignment2_Report.pdf` | Full academic report (24 pages) detailing methodology and findings |
| `titanic-dataset.csv` | Original dataset (891 passengers, 12 features) |
| `fig1_loss_curve.png` | ANN training loss over epochs |
| `fig2_confusion_matrices.png` | Confusion matrices for all three models |
| `fig3_roc_curves.png` | ROC curves with AUC scores |
| `fig4_metrics_comparison.png` | Bar chart comparing all performance metrics |

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| ANN (MLP) | ~0.82 | ~0.80 | ~0.75 | ~0.77 | ~0.87 |
| Logistic Regression | ~0.81 | ~0.79 | ~0.74 | ~0.76 | ~0.86 |
| Decision Tree | ~0.79 | ~0.74 | ~0.74 | ~0.74 | ~0.83 |

The ANN achieved the highest overall performance, demonstrating the value of non-linear modeling for this classification task.

## Technologies Used

- Python 3.10
- pandas, numpy
- scikit-learn (MLPClassifier, LogisticRegression, DecisionTreeClassifier, StandardScaler, metrics)
- matplotlib, seaborn

## How to Run

1. Clone this repository
2. Install required libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the Jupyter notebook: `jupyter notebook Assignment2_Titanic.ipynb`

## Author

[**Andile Gift Shabalala**](https://www.linkedin.com/in/andile-gift-shabalala-2345bb313/) 

## Acknowledgments

- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- Built upon EDA work completed in Assignment 1 [GitHub](https://github.com/andilegift/ISM8X04-Learning-from-Data-Assignment )

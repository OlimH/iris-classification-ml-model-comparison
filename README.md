# Iris Species Classification: Machine Learning Model Comparison

## Project Overview

This project implements an end-to-end supervised machine learning workflow to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.

Three classification algorithms are developed, optimized, and compared:

- Logistic Regression
- Random Forest
- XGBoost

The project covers exploratory data analysis, data preprocessing, feature selection, hyperparameter optimization, cross-validation, threshold analysis, model evaluation, and feature importance.

**Objective:** Identify the best-performing classification model while balancing predictive accuracy, model complexity, and interpretability.

## Dataset

The Iris dataset contains 150 observations distributed equally across three flower species.

**Predictor variables:**

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

**Target variable:** Iris species (Setosa, Versicolor, Virginica).

Each species contains 50 observations, making the dataset perfectly balanced.

An 80/20 stratified train-test split is used to preserve class proportions.

## Machine Learning Workflow

### 1. Exploratory Data Analysis (EDA)

- Examine dataset structure and descriptive statistics.
- Analyze target class distribution.
- Explore feature distributions and relationships.
- Investigate feature correlations.
- Visualize species separation.

### 2. Data Preprocessing

- Separate predictors and target variables.
- Apply stratified train-test splitting.
- Scale numerical features where appropriate.
- Use preprocessing pipelines to prevent data leakage.
- Preserve the test dataset for final evaluation.

Because the dataset is balanced, oversampling methods such as SMOTE are not required.

### 3. Feature Selection

Three feature selection approaches are compared:

- All four original features (baseline)
- Recursive Feature Elimination (RFE)
- L1 regularization-based feature selection

All three approaches achieved approximately 0.958 mean cross-validation F1-macro, suggesting that some features may be removed without materially affecting predictive performance.

### 4. Model Development and Hyperparameter Tuning

Three classification algorithms are trained and optimized using cross-validation.

| Model | Tuned CV F1-Macro |
|---|---:|
| Logistic Regression | 0.966 |
| Random Forest | 0.966 |
| XGBoost | 0.966 |

Hyperparameters explored include regularization strength, maximum tree depth, feature sampling, learning rate, number of estimators, and subsampling.

### 5. Cross-Validation Model Comparison

Models are evaluated using out-of-fold predictions and multiple performance metrics:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- ROC-AUC

All models achieved approximately 96.7% out-of-fold accuracy.

Logistic Regression achieved the highest macro one-vs-rest ROC-AUC of 0.997, followed by Random Forest (0.996) and XGBoost (0.994).

### 6. ROC and Precision-Recall Analysis

Micro-averaged ROC and Precision-Recall curves are generated to assess classification performance across probability thresholds.

| Model | Micro ROC-AUC | Micro Average Precision |
|---|---:|---:|
| Logistic Regression | 0.998 | 0.996 |
| Random Forest | 0.998 | 0.995 |
| XGBoost | 0.996 | 0.993 |

All three models demonstrate excellent class discrimination.

### 7. Classification Threshold Analysis

Threshold tuning is performed for Virginica versus the remaining species.

Probability thresholds ranging from 0.10 to 0.90 are evaluated using precision, recall, and F1-score.

Thresholds between 0.15 and 0.25 achieved an F1-score of 0.952, compared with 0.950 at the default threshold of 0.50.

The default threshold remains a reasonable choice because it provides a more balanced precision-recall trade-off.

### 8. Final Model Evaluation

Logistic Regression is selected based on its strong cross-validation performance, simplicity, and interpretability.

The selected model is evaluated on the untouched test dataset containing 30 observations.

| Metric | Test Result |
|---|---:|
| Accuracy | 1.000 |
| Macro Precision | 1.000 |
| Macro Recall | 1.000 |
| Macro F1-score | 1.000 |
| Macro ROC-AUC | 1.000 |

The model correctly classified all 30 test observations, with no misclassifications.

Although the results are excellent, the relatively small test sample means perfect performance cannot be guaranteed on future unseen data.

### 9. Feature Importance and Interpretability

Permutation feature importance is used to evaluate how much each predictor contributes to model performance.

The results identify petal measurements as the strongest predictors.

| Feature | Approximate Importance |
|---|---:|
| Petal Width | 0.27 |
| Petal Length | 0.26 |
| Sepal Width | 0.10 |
| Sepal Length | 0.08 |

These results indicate that petal measurements provide the most useful information for distinguishing between Iris species.

Permutation importance on the test dataset is used for diagnostic interpretation only, not for further model optimization.

## Key Findings

- All three classification models achieved strong predictive performance.
- Feature selection showed that reducing the number of predictors can preserve model performance.
- Hyperparameter tuning produced similar cross-validation scores across the three algorithms.
- Logistic Regression provided excellent predictive performance with lower model complexity.
- The final Logistic Regression model correctly classified all 30 held-out test observations.
- Petal width and petal length were the most influential predictors.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Running the Project

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/iris-classification-ml-model-comparison.git
cd iris-classification-ml-model-comparison
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

`notebooks/iris_classification_model_comparison.ipynb`

Run the notebook cells sequentially to reproduce the analysis.

## Conclusion

This project demonstrates a complete multiclass machine learning classification workflow, including preprocessing, feature selection, hyperparameter optimization, model comparison, and interpretability.

Logistic Regression was selected as the preferred model because it achieved excellent classification performance while remaining simple, computationally efficient, and interpretable.

The project highlights the importance of evaluating multiple algorithms and selecting models based on validation performance and practical considerations rather than model complexity alone.

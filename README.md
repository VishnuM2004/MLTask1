# Iris Multi-Class Classification Benchmark

## Project Overview
This project compares two supervised machine learning algorithms, Logistic Regression and Decision Tree, using the built-in Iris dataset from Scikit-Learn.

The objective is to evaluate and compare classification performance using standard machine learning evaluation metrics.

## Technologies Used
- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Dataset
The Iris dataset contains 150 samples belonging to three flower species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

It uses four input features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Machine Learning Algorithms
1. Logistic Regression – uses StandardScaler within a pipeline for feature scaling.
2. Decision Tree Classifier – classifies samples using decision-based rules.

## Methodology
1. Load the Iris dataset.
2. Split the dataset into 80% training and 20% testing using stratified sampling.
3. Apply feature scaling for Logistic Regression.
4. Train both classification models.
5. Evaluate model performance.
6. Generate confusion matrices and visualizations.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Project Structure
```text
iris_classification/
├── README.md
├── classification_pipeline.py
├── requirements.txt
└── outputs/
    ├── benchmark_metrics.csv
    ├── iris_pairplot.png
    ├── decision_tree_confusion_matrix.png
    └── logistic_regression_confusion_matrix.png
```

## Installation and Execution

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the project:

```bash
python classification_pipeline.py
```

## Output
The project generates:
- Model performance comparison in CSV format.
- Confusion matrices for both models.
- Iris dataset pairplot visualization.
- Classification reports.

## Conclusion
This project demonstrates an end-to-end multi-class classification pipeline and compares Logistic Regression and Decision Tree algorithms using the Iris dataset.

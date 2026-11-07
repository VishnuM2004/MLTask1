# Iris multi-class classification benchmark

Compares Logistic Regression and a Decision Tree on Scikit-Learn's built-in Iris dataset. The split is stratified 80/20 with a fixed random seed. The Logistic Regression model scales features with `StandardScaler` inside a pipeline, which prevents test-set information leaking into the scaler. Decision Trees do not require scaling, so that model uses the original measurements.

## Run in VS Code

Open this folder in VS Code. In its terminal, create and activate a virtual environment, install packages, then run the script:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python classification_pipeline.py
```

On macOS/Linux activate with `source .venv/bin/activate`.

The script prints per-class reports and macro-averaged accuracy, precision, recall, and F1. It saves `benchmark_metrics.csv`, a pairplot, one confusion matrix per model, and `decision_boundaries.png` under the `outputs` folder.

The decision boundary chart intentionally trains a second pair of models on petal length and petal width so the multi-class regions can be shown in two dimensions. Treat that chart as a visualization; the benchmark metrics use all four Iris features.

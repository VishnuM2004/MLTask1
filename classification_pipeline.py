"""Compare Logistic Regression and Decision Tree on the Iris benchmark."""
from pathlib import Path
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs", help="Folder for plots and metrics")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bunch = load_iris(as_frame=True)
    data = bunch.frame
    data["species"] = data["target"].map(dict(enumerate(bunch.target_names)))
    X = data[bunch.feature_names]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    sns.pairplot(data, vars=bunch.feature_names, hue="species", corner=True)
    plt.savefig(out / "iris_pairplot.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        # Tree models do not need scaling. Keeping raw values avoids implying otherwise.
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        })
        print(f"\n{name}\n" + classification_report(y_test, pred, target_names=bunch.target_names, zero_division=0))
        cm = confusion_matrix(y_test, pred, labels=range(3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=bunch.target_names, yticklabels=bunch.target_names)
        plt.title(f"{name} confusion matrix")
        plt.xlabel("Predicted species")
        plt.ylabel("True species")
        plt.tight_layout()
        filename = name.lower().replace(" ", "_")
        plt.savefig(out / f"{filename}_confusion_matrix.png", dpi=160)
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(out / "benchmark_metrics.csv", index=False)
    print("\nBenchmark metrics (macro averages):")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    # For a readable 2D boundary, fit each model on the two most discriminative
    # features (petal length and width). The test split remains the same.
    boundary_features = ["petal length (cm)", "petal width (cm)"]
    x2_train, x2_test = X_train[boundary_features], X_test[boundary_features]
    x_min, x_max = X[boundary_features[0]].min() - .5, X[boundary_features[0]].max() + .5
    y_min, y_max = X[boundary_features[1]].min() - .5, X[boundary_features[1]].max() + .5
    xx, yy = __import__("numpy").meshgrid(
        __import__("numpy").arange(x_min, x_max, .02),
        __import__("numpy").arange(y_min, y_max, .02),
    )
    grid = pd.DataFrame({boundary_features[0]: xx.ravel(), boundary_features[1]: yy.ravel()})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, (name, original_model) in zip(axes, models.items()):
        if name == "Logistic Regression":
            boundary_model = Pipeline([("scaler", StandardScaler()),
                                       ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
        else:
            boundary_model = DecisionTreeClassifier(random_state=42, max_depth=4)
        boundary_model.fit(x2_train, y_train)
        regions = boundary_model.predict(grid).to_numpy().reshape(xx.shape)
        ax.contourf(xx, yy, regions, alpha=.25, cmap="viridis")
        ax.scatter(x2_test.iloc[:, 0], x2_test.iloc[:, 1], c=y_test,
                   cmap="viridis", edgecolor="black", s=45)
        ax.set_title(f"{name} (2 features)")
        ax.set_xlabel(boundary_features[0])
        ax.set_ylabel(boundary_features[1])
    fig.suptitle("Decision boundaries on petal measurements (test points)")
    fig.tight_layout()
    fig.savefig(out / "decision_boundaries.png", dpi=160)
    plt.close(fig)
    print(f"\nPlots and metrics saved in: {out.resolve()}")


if __name__ == "__main__":
    main()

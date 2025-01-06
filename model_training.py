import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Data exploration
print("Dataset Description:\n", data.DESCR[:500])
print("\nFirst 5 rows of the dataset:")
print(pd.DataFrame(X, columns=feature_names).head())

# Perform PCA for dimensionality reduction
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Explained variance by components
print("\nExplained Variance Ratio by PCA Components:", pca.explained_variance_ratio_)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

# Define classifiers with hyperparameter tuning
classifiers = {
    "Logistic Regression": GridSearchCV(LogisticRegression(), param_grid={"C": [0.1, 1, 10]}, cv=5),
    "Support Vector Machine": GridSearchCV(SVC(), param_grid={"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}, cv=5),
    "Random Forest": GridSearchCV(RandomForestClassifier(), param_grid={"n_estimators": [50, 100, 150]}, cv=5),
    "Decision Tree": GridSearchCV(DecisionTreeClassifier(), param_grid={"max_depth": [3, 5, 10]}, cv=5),
    "K-Nearest Neighbors": GridSearchCV(KNeighborsClassifier(), param_grid={"n_neighbors": [3, 5, 7]}, cv=5),
    "Naive Bayes": GaussianNB(),
    "Artificial Neural Network": GridSearchCV(MLPClassifier(max_iter=500), param_grid={"hidden_layer_sizes": [(50,), (100,), (50, 50)]}, cv=5)
}

# Create pipeline
pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("classifier", None)
])

# Train and evaluate all models
results = {"Accuracy": {}, "Precision": {}, "Recall": {}, "F1-Score": {}}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    pipeline.set_params(classifier=clf)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    results["Accuracy"][name] = accuracy_score(y_test, y_pred)
    results["Precision"][name] = precision_score(y_test, y_pred)
    results["Recall"][name] = recall_score(y_test, y_pred)
    results["F1-Score"][name] = f1_score(y_test, y_pred)

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print("\nModel Performance Metrics:")
print(results_df)

# Plot: Comparison of Accuracy, Precision, Recall, and F1-Score
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
bar_width = 0.2
x = np.arange(len(classifiers))

plt.figure(figsize=(14, 8))
for i, metric in enumerate(metrics_to_plot):
    plt.bar(x + i * bar_width, results_df[metric], bar_width, label=metric)

plt.xlabel("Models")
plt.ylabel("Scores")
plt.title("Comparison of Metrics Across Algorithms")
plt.xticks(x + bar_width * 1.5, classifiers.keys(), rotation=45, ha="right")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Confusion Matrix for the Best Model (based on F1-Score)
best_model_name = results_df["F1-Score"].idxmax()
best_model = classifiers[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Retrain best model on full training data
pipeline.set_params(classifier=best_model)
pipeline.fit(X_train, y_train)
y_best_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_best_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"])
plt.title(f"Confusion Matrix for {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(f"\nClassification Report for {best_model_name}:")
print(classification_report(y_test, y_best_pred))

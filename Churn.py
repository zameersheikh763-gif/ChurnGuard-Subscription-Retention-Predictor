import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay

#  Load dataset
df = pd.read_csv("/content/subscription_data.csv")

#  Missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

#  Encode Plan_Type
encoder = LabelEncoder()
df["Plan_Type"] = encoder.fit_transform(df["Plan_Type"])

#  Features and target
X = df.drop(["User_ID", "Churned"], axis=1)
y = df["Churned"]

#  Better split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#  Optimized model
best_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42
)

best_model.fit(X_train, y_train)

# Prediction
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

#  Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("AUC ROC:", auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance graph
importance = pd.Series(best_model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", figsize=(8, 5))
plt.title("Top Churn Factors")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

#  Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()

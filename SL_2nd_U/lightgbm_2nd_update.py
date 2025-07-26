import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
import shap
import matplotlib.pyplot as plt

# Set paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../SL_2nd_U_results/lightgbm_2nd_updated_results"
os.makedirs(results_dir, exist_ok=True)

print(" Loading labeled dataset from:", data_path)
df = pd.read_csv(data_path, low_memory=False)
df = df[df["MASH_LABEL"].isin([0, 1])]

# Optional structured features to add
structured_features = ["age", "gender"]  # Extend as needed

for model_name in ["biobert", "clinicalbert", "pubmedbert"]:
    print(f"\n Processing model: {model_name}")
    embed_path = os.path.join(embeddings_dir, f"final_embeddings_{model_name}.npy")
    embeddings = np.load(embed_path)

    if len(embeddings) != len(df):
        print(f" Mismatch in length. Truncating to {min(len(embeddings), len(df))} rows.")
        df = df.iloc[:len(embeddings)]
        embeddings = embeddings[:len(df)]

    # Combine embeddings with tabular data
    if all(feat in df.columns for feat in structured_features):
        tabular = df[structured_features].values
        X = np.hstack([embeddings, tabular])
    else:
        X = embeddings

    y = df["MASH_LABEL"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = LGBMClassifier(
        objective="binary",
        device="gpu",
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.01,
        num_leaves=64,
        lambda_l1=0.1,
        lambda_l2=0.1,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=1000
    )

    print(f" Training LGBMClassifier for {model_name}...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)]
    )

    y_pred_probs = model.predict_proba(X_test)[:, 1]

    # Threshold tuning
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    y_pred_binary = (y_pred_probs > best_threshold).astype(int)

    # Metrics
    accuracy = np.mean(y_pred_binary == y_test)
    auc_roc = roc_auc_score(y_test, y_pred_probs)
    auc_pr = average_precision_score(y_test, y_pred_probs)
    report = classification_report(y_test, y_pred_binary, digits=4)
    cm = confusion_matrix(y_test, y_pred_binary)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    shap_path = os.path.join(results_dir, f"{model_name}_shap_summary.png")
    plt.savefig(shap_path)
    plt.close()

    # Save report
    result_text = (
        f"Model: {model_name}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"AUC-ROC: {auc_roc:.4f}\n"
        f"AUC-PR: {auc_pr:.4f}\n"
        f"Best Threshold (F1-optimized): {best_threshold:.4f}\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}\n"
        f"SHAP Summary Plot Saved: {shap_path}"
    )

    result_file = os.path.join(results_dir, f"{model_name}_lightgbm_results.txt")
    with open(result_file, "w") as f:
        f.write(result_text)

    print(f" Results saved to: {result_file}")
    print(f" SHAP summary plot saved to: {shap_path}")

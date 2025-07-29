import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

# Paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../SL_3rd_U_results/xgboost_3rd_updated_results"
os.makedirs(results_dir, exist_ok=True)

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path, low_memory=False)
df = df[df["MASH_LABEL"].isin([0, 1])]

structured_features = ["age", "gender"]

models = {
    "biobert": "final_embeddings_biobert.npy",
    "clinicalbert": "final_embeddings_clinicalbert.npy",
    "pubmedbert": "final_embeddings_pubmedbert.npy"
}

for model_name, embedding_file in models.items():
    print(f"\nProcessing model: {model_name}")
    embed_path = os.path.join(embeddings_dir, embedding_file)
    embeddings = np.load(embed_path)

    if len(embeddings) != len(df):
        min_len = min(len(embeddings), len(df))
        print(f" Truncating to {min_len} rows.")
        df = df.iloc[:min_len]
        embeddings = embeddings[:min_len]

    if all(f in df.columns for f in structured_features):
        tabular = df[structured_features].values
        X = np.hstack([embeddings, tabular])
    else:
        X = embeddings

    y = df["MASH_LABEL"].values

    # SMOTE balancing
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    clf = xgb.XGBClassifier(
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbosity=1,
        use_label_encoder=False
    )

    print(f" Training XGBoost model for {model_name}...")
    clf.fit(X_train, y_train)

    y_pred_probs = clf.predict_proba(X_test)[:, 1]

    # F1-based thresholding
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    y_pred = (y_pred_probs > best_threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_probs)
    auc_pr = average_precision_score(y_test, y_pred_probs)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # SHAP Explanation
    explainer = shap.Explainer(clf)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    shap_path = os.path.join(results_dir, f"{model_name}_shap_summary.png")
    plt.savefig(shap_path)
    plt.close()

    # Save results
    output_file = os.path.join(results_dir, f"{model_name}_xgboost_results.txt")
    with open(output_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"AUC-PR: {auc_pr:.4f}\n")
        f.write(f"Best Threshold (F1-optimized): {best_threshold:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")
        f.write(f"SHAP Summary Plot Saved: {shap_path}\n")

    print(f" Results saved to: {output_file}")
    print(f" SHAP summary plot saved to: {shap_path}")

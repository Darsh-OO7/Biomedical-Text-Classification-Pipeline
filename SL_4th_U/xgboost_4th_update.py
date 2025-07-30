import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)
from imblearn.under_sampling import RandomUnderSampler
import shap
import matplotlib.pyplot as plt

# Paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../SL_4th_U_results/xgboost_4th_updated_results"
os.makedirs(results_dir, exist_ok=True)

print("Loading dataset from:", data_path)
df = pd.read_csv(data_path, low_memory=False)
df = df[df["MASH_LABEL"].isin([0, 1])]
structured_features = ["age", "gender"]

for model_name in ["biobert", "clinicalbert", "pubmedbert"]:
    print(f"\nProcessing model: {model_name}")
    embed_path = os.path.join(embeddings_dir, f"final_embeddings_{model_name}.npy")
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

    # Custom split: under-sample train only, keep val/test imbalanced
    X_train_full, X_temp, y_train_full, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_train_full, y_train_full)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        use_label_encoder=False,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        gpu_id=0,
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=1000
    )

    print(f" Training XGBoostClassifier for {model_name} on GPU...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=100)

    y_pred_probs = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    y_pred_binary = (y_pred_probs > best_threshold).astype(int)

    accuracy = np.mean(y_pred_binary == y_val)
    auc_roc = roc_auc_score(y_val, y_pred_probs)
    auc_pr = average_precision_score(y_val, y_pred_probs)
    report = classification_report(y_val, y_pred_binary, digits=4)
    cm = confusion_matrix(y_val, y_pred_binary)

    # SHAP analysis (fix)
    if isinstance(X_val, np.ndarray):
        X_val_df = pd.DataFrame(X_val)
    else:
        X_val_df = X_val

    shap_path = os.path.join(results_dir, f"{model_name}_shap_summary.png")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_df)
        shap.summary_plot(shap_values, X_val_df, show=False)
        plt.savefig(shap_path)
        plt.close()
        shap_note = f"SHAP Summary Plot Saved: {shap_path}"
    except Exception as e:
        shap_note = f"SHAP Plot Skipped due to error: {e}"
        print(shap_note)

    result_text = (
        f"Model: {model_name}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"AUC-ROC: {auc_roc:.4f}\n"
        f"AUC-PR: {auc_pr:.4f}\n"
        f"Best Threshold (F1-optimized): {best_threshold:.4f}\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}\n"
        f"{shap_note}"
    )

    result_file = os.path.join(results_dir, f"{model_name}_xgboost_results.txt")
    with open(result_file, "w") as f:
        f.write(result_text)

    print(f" Results saved to: {result_file}")

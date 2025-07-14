# Enhanced XGBoost Training Script with Class Weighting and Threshold Tuning
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Set file paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../supervised_learning_updated_results/xgboost_updated_results"
os.makedirs(results_dir, exist_ok=True)

# Load dataset
print(f" Loading labeled dataset from: {data_path}")
df = pd.read_csv(data_path, low_memory=False)
df = df[df["MASH_LABEL"].isin([0, 1])]

models = {
    "biobert": "final_embeddings_biobert.npy",
    "clinicalbert": "final_embeddings_clinicalbert.npy",
    "pubmedbert": "final_embeddings_pubmedbert.npy"
}

for model_name, embedding_file in models.items():
    print(f"\n Processing model: {model_name}")

    embedding_path = os.path.join(embeddings_dir, embedding_file)
    embeddings = np.load(embedding_path)

    if len(embeddings) != len(df):
        min_len = min(len(embeddings), len(df))
        print(f" Mismatch in length. Truncating to {min_len} rows.")
        X = embeddings[:min_len]
        y = df["MASH_LABEL"].values[:min_len]
    else:
        X = embeddings
        y = df["MASH_LABEL"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Compute class weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    clf = xgb.XGBClassifier(
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        verbosity=1
    )

    print(f" Training XGBoost model for {model_name}...")
    clf.fit(X_train, y_train)

    # Predict probabilities and apply threshold tuning
    y_pred_probs = clf.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred = (y_pred_probs > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    output_file = os.path.join(results_dir, f"{model_name}_xgboost_results.txt")
    with open(output_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Threshold: {threshold}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))

    print(f" Results saved to {output_file}")

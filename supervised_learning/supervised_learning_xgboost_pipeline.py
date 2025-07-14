import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set file paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../supervised_learning_results/xgboost_results"

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Load dataset
print(f" Loading labeled dataset from: {data_path}")
df = pd.read_csv(data_path, low_memory=False)

# Use correct label column
label_column = "MASH_LABEL"
if label_column not in df.columns:
    raise KeyError(f" Column '{label_column}' not found in dataset.")

# Confirm binary classification task
df = df[df[label_column].isin([0, 1])]

# Define models to process
models = {
    "biobert": "final_embeddings_biobert.npy",
    "clinicalbert": "final_embeddings_clinicalbert.npy",
    "pubmedbert": "final_embeddings_pubmedbert.npy"
}

# Iterate over all embeddings
for model_name, embedding_file in models.items():
    print(f"\n Processing model: {model_name}")
    
    # Load embeddings
    embedding_path = os.path.join(embeddings_dir, embedding_file)
    embeddings = np.load(embedding_path)
    print(f" Loaded embeddings with shape: {embeddings.shape}")

    # Align embeddings with the dataset length
    if len(embeddings) != len(df):
        min_len = min(len(embeddings), len(df))
        print(f" Mismatch in length. Truncating to {min_len} rows.")
        X = embeddings[:min_len]
        y = df[label_column].values[:min_len]
    else:
        X = embeddings
        y = df[label_column].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Define XGBoost classifier with GPU acceleration
    clf = xgb.XGBClassifier(
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=1
    )

    # Train model
    print(f" Training XGBoost model for {model_name}...")
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save results
    output_file = os.path.join(results_dir, f"{model_name}_xgboost_results.txt")
    with open(output_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))

    print(f"Results saved to {output_file}")

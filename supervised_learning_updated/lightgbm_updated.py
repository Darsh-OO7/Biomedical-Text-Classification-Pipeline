# Enhanced LightGBM Training Script with Class Weighting and Threshold Tuning
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Set paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../supervised_learning_updated_results/lightgbm_updated_results"
os.makedirs(results_dir, exist_ok=True)

print("\U0001F4C4 Loading labeled dataset from:", data_path)
df = pd.read_csv(data_path)
df = df[df['MASH_LABEL'].isin([0, 1])]

for model_name in ["biobert", "clinicalbert", "pubmedbert"]:
    print(f"\n\U0001F50D Processing model: {model_name}")
    embed_path = os.path.join(embeddings_dir, f"final_embeddings_{model_name}.npy")
    embeddings = np.load(embed_path)

    if len(embeddings) != len(df):
        print(f" Mismatch in length. Truncating to {min(len(embeddings), len(df))} rows.")
        df = df.iloc[:len(embeddings)]
        embeddings = embeddings[:len(df)]

    X = embeddings
    y = df["MASH_LABEL"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # LightGBM parameters with class weighting
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "device": "gpu",
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    print(f"\U0001F680 Training LightGBM model for {model_name}...")
    model = lgb.train(params, train_data, num_boost_round=100)

    # Predict probabilities and apply threshold tuning
    y_pred_probs = model.predict(X_test)
    threshold = 0.3
    y_pred_binary = (y_pred_probs > threshold).astype(int)

    # Evaluation
    report = classification_report(y_test, y_pred_binary, digits=4)
    cm = confusion_matrix(y_test, y_pred_binary)
    accuracy = np.mean(y_pred_binary == y_test)
    auc = roc_auc_score(y_test, y_pred_probs)

    result_text = (
        f"Model: {model_name}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"AUC: {auc:.4f}\n"
        f"Threshold: {threshold}\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}"
    )

    result_file = os.path.join(results_dir, f"{model_name}_lightgbm_results.txt")
    with open(result_file, "w") as f:
        f.write(result_text)

    print(f" Results saved to: {result_file}")

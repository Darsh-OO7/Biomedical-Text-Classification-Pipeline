import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)
from imblearn.under_sampling import RandomUnderSampler

# Paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../SL_4th_U_results/pytorchnn_4th_updated_results"
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv(data_path, low_memory=False)
df = df[df["MASH_LABEL"].isin([0, 1])]
structured_features = ["age", "gender"]

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

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

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    pos_weight = (y_train_tensor == 0).sum() / (y_train_tensor == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    model = DeepNN(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f" Training PyTorch NN model for {model_name}...")
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f" Epoch [{epoch+1}/20], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_val_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        y_pred = (probs > best_threshold).astype(int)

    accuracy = np.mean(y_pred == y_val_tensor.cpu().numpy())
    auc_roc = roc_auc_score(y_val, probs)
    auc_pr = average_precision_score(y_val, probs)
    report = classification_report(y_val, y_pred, digits=4)
    cm = confusion_matrix(y_val, y_pred)

    result_text = (
        f"Model: {model_name}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"AUC-ROC: {auc_roc:.4f}\n"
        f"AUC-PR: {auc_pr:.4f}\n"
        f"Best Threshold (F1-optimized): {best_threshold:.4f}\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}"
    )

    result_file = os.path.join(results_dir, f"{model_name}_pytorch_nn_results.txt")
    with open(result_file, "w") as f:
        f.write(result_text)

    print(f" Results saved to: {result_file}")

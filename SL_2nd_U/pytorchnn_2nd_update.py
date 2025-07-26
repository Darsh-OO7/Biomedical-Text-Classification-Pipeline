import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

# Set paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../SL_2nd_U_results/pytorchnn_2nd_updated_results"
os.makedirs(results_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# Load and filter dataset
print(" Loading labeled dataset from:", data_path)
df = pd.read_csv(data_path, low_memory=False)
df = df[df["MASH_LABEL"].isin([0, 1])]

# Optional structured features to fuse with embeddings
structured_features = ["age", "gender"]  

# Define model architecture
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedForwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Loop through each embedding model
for model_name in ["biobert", "clinicalbert", "pubmedbert"]:
    print(f"\n Processing model: {model_name}")
    embed_path = os.path.join(embeddings_dir, f"final_embeddings_{model_name}.npy")
    embeddings = np.load(embed_path)

    if len(embeddings) != len(df):
        print(f" Mismatch in length. Truncating to {min(len(embeddings), len(df))} rows.")
        df = df.iloc[:len(embeddings)]
        embeddings = embeddings[:len(df)]

    # Concatenate embeddings + structured features
    if all(f in df.columns for f in structured_features):
        tabular = df[structured_features].values
        X = np.hstack([embeddings, tabular])
    else:
        X = embeddings

    y = df["MASH_LABEL"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

    # Class weighting
    scale_pos_weight = (y_train_tensor == 0).sum().item() / (y_train_tensor == 1).sum().item()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(scale_pos_weight, dtype=torch.float32).to(device))

    # Model and optimizer
    model = FeedForwardNN(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    print(f" Training PyTorch NN model for {model_name}...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f" Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_pred_probs = torch.sigmoid(logits)

        # F1-based threshold tuning
        y_test_np = y_test_tensor.cpu().numpy()
        probs_np = y_pred_probs.cpu().numpy()
        precision, recall, thresholds = precision_recall_curve(y_test_np, probs_np)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        y_pred = (y_pred_probs > best_threshold).float()

    # Metrics
    y_pred_np = y_pred.cpu().numpy()
    accuracy = np.mean(y_pred_np == y_test_np)
    auc_roc = roc_auc_score(y_test_np, probs_np)
    auc_pr = average_precision_score(y_test_np, probs_np)
    report = classification_report(y_test_np, y_pred_np, digits=4)
    cm = confusion_matrix(y_test_np, y_pred_np)

    # Save results
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

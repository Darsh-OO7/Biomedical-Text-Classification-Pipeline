# Enhanced PyTorch NN Training Script with Class Weighting and Threshold Tuning
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np

# Paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../supervised_learning_updated_results/pytorchnn_updated_results"
os.makedirs(results_dir, exist_ok=True)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Load labeled data
print("📄 Loading labeled dataset from:", data_path)
df = pd.read_csv(data_path)
df = df[df['MASH_LABEL'].isin([0, 1])]

# Feedforward Neural Network
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

for model_name in ["biobert", "clinicalbert", "pubmedbert"]:
    print(f"\n🔍 Processing model: {model_name}")
    embed_path = os.path.join(embeddings_dir, f"final_embeddings_{model_name}.npy")
    embeddings = np.load(embed_path)

    if len(embeddings) != len(df):
        min_len = min(len(embeddings), len(df))
        print(f"⚠️ Mismatch in length. Truncating to {min_len} rows.")
        df = df.iloc[:min_len]
        embeddings = embeddings[:min_len]

    X = embeddings
    y = df["MASH_LABEL"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

    # Compute class weights
    scale_pos_weight = (y_train == 0).sum().item() / (y_train == 1).sum().item()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(scale_pos_weight, dtype=torch.float32).to(device))

    # Initialize model
    model = FeedForwardNN(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"🚀 Training PyTorch NN model for {model_name}...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f"📉 Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        y_pred_probs = torch.sigmoid(logits)
        threshold = 0.3
        y_pred = (y_pred_probs > threshold).float()

    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    y_pred_probs_np = y_pred_probs.cpu().numpy()

    report = classification_report(y_test_np, y_pred_np, digits=4)
    cm = confusion_matrix(y_test_np, y_pred_np)
    accuracy = np.mean(y_pred_np == y_test_np)
    auc = roc_auc_score(y_test_np, y_pred_probs_np)

    result_text = (
        f"Model: {model_name}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"AUC: {auc:.4f}\n"
        f"Threshold: {threshold}\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}"
    )

    result_file = os.path.join(results_dir, f"{model_name}_pytorch_nn_results.txt")
    with open(result_file, "w") as f:
        f.write(result_text)

    print(f"✅ Results saved to: {result_file}")

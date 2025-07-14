import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set paths
data_path = "../embeddings/final_patient_dataset_cleaned.csv"
embeddings_dir = "../embeddings/embeddings_output"
results_dir = "../supervised_learning_results/lightgbm_results"
os.makedirs(results_dir, exist_ok=True)

print("📄 Loading labeled dataset from:", data_path)
df = pd.read_csv(data_path)
df = df[df['MASH_LABEL'].isin([0, 1])]

# Loop through embeddings
for model_name in ["biobert", "clinicalbert", "pubmedbert"]:
    print(f"\n🔍 Processing model: {model_name}")
    embed_path = os.path.join(embeddings_dir, f"final_embeddings_{model_name}.npy")
    embeddings = np.load(embed_path)

    # Align length
    if len(embeddings) != len(df):
        print(f"⚠️ Mismatch in length. Truncating to {min(len(embeddings), len(df))} rows.")
        df = df.iloc[:len(embeddings)]
        embeddings = embeddings[:len(df)]

    X = embeddings
    y = df["MASH_LABEL"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"🚀 Training LightGBM model for {model_name}...")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Parameters for GPU training
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "device": "gpu",
        "verbosity": -1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100
    )

    # Predict
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    report = classification_report(y_test, y_pred_binary, digits=4)
    cm = confusion_matrix(y_test, y_pred_binary)
    accuracy = np.mean(y_pred_binary == y_test)

    result_text = (
        f"Model: {model_name}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}"
    )

    result_file = os.path.join(results_dir, f"{model_name}_lightgbm_results.txt")
    with open(result_file, "w") as f:
        f.write(result_text)

    print(f"✅ Results saved to: {result_file}")

    class MASHNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)  


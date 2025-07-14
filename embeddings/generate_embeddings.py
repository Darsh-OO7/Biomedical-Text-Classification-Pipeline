import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import Counter
import os

# -------------------- Configuration --------------------
DATA_PATH = "final_patient_dataset_cleaned.csv"
OUTPUT_DIR = "embeddings_output"
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "biobert": "dmis-lab/biobert-v1.1"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- Load Dataset --------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
texts = df['CLEAN_TEXT'].dropna().tolist()

# -------------------- Embedding Function --------------------
def generate_embeddings(model_name, model_path):
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(DEVICE)
    model.eval()

    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Embedding with {model_name}"):
        batch_texts = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings)
    print(f"\n{model_name} embeddings shape: {embeddings.shape}")

    # Save embeddings
    save_path = os.path.join(OUTPUT_DIR, f"final_embeddings_{model_name}.npy")
    np.save(save_path, embeddings)
    print(f"Saved embeddings to {save_path}")

# -------------------- Run for All Models --------------------
for model_name, model_path in MODELS.items():
    generate_embeddings(model_name, model_path)

print("\nAll embeddings generated and saved.")

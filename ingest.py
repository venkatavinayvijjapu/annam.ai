# ingest.py
# Step 1: Load and preprocess data
# Step 2: Generate embeddings and store in ChromaDB with batching

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# --------------------------
# CONFIGURATION
# --------------------------
CSV_PATH = "questionsv4.csv"
CHROMA_DIR = "./kcc_chroma"
COLLECTION_NAME = "kcc_qna"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 5000  # Adjust as needed, max ~5461 for your environment

# --------------------------
# STEP 1: Load & Clean CSV
# --------------------------
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["questions", "answers"]).reset_index(drop=True)
df = df.rename(columns={"questions": "question", "answers": "answer"})
df.to_csv("kcc_cleaned.csv", index=False)  # Save cleaned version

# --------------------------
# STEP 2: Generate Embeddings
# --------------------------
model = SentenceTransformer(EMBED_MODEL_NAME)
embeddings = model.encode(df["question"].tolist(), show_progress_bar=True, normalize_embeddings=True).tolist()

# --------------------------
# STEP 3: Store in ChromaDB (with batching)
# --------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# Optional: clear existing collection to avoid duplicates
all_data = collection.get()
all_ids = all_data.get("ids", [])

if len(all_ids) > 0:
    collection.delete(ids=all_ids)

def batch_add(collection, docs, embeds, metadatas, ids, batch_size=5000):
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i+batch_size],
            embeddings=embeds[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

batch_add(
    collection,
    docs=df["answer"].tolist(),
    embeds=embeddings,
    metadatas=[{"question": q} for q in df["question"]],
    ids=[str(i) for i in range(len(df))],
    batch_size=BATCH_SIZE
)

print(f"Ingested {len(df)} Q&A pairs into ChromaDB collection '{COLLECTION_NAME}'.")


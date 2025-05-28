import os
import sys
import pathlib
import pandas as pd
from sentence_transformers import SentenceTransformer
from weaviate import Client, AuthApiKey
from dotenv import load_dotenv

# ------------------------------------------------------------------
# 0  Config & model
# ------------------------------------------------------------------
load_dotenv()
CSV_PATH   = pathlib.Path("resources.csv")
CLASS_NAME = "AidProgram"
MODEL_NAME = "all-MiniLM-L6-v2"
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

print("📥 Loading embedding model… (this can take ~30 s on first run)")
embedder = SentenceTransformer(MODEL_NAME)
print("✅ Model loaded.")

# ------------------------------------------------------------------
# 1  Connect to Weaviate (cloud auth)
# ------------------------------------------------------------------
client = Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API_KEY)
)

# ------------------------------------------------------------------
# 2  Schema + data upload
# ------------------------------------------------------------------
aid_schema = {
    "class": CLASS_NAME,
    "description": "Local aid programs (manual vectors)",
    "vectorizer": "none",
    "properties": [
        {"name": "name", "dataType": ["text"]},
        {"name": "type", "dataType": ["text"]},
        {"name": "city", "dataType": ["text"]},
        {"name": "description", "dataType": ["text"]},
        {"name": "contact", "dataType": ["text"]},
    ],
}

if not client.schema.contains(aid_schema):
    client.schema.create_class(aid_schema)

print("📄 Loading resources.csv…")
df = pd.read_csv(CSV_PATH)
current = client.query.aggregate(CLASS_NAME).with_meta_count().do()
count   = current["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"]
if count == 0:
    print("🔧 Vectorising & uploading", len(df), "rows → Weaviate…")
    vectors = embedder.encode(df["description"].tolist(), batch_size=64, show_progress_bar=True)
    for row, vec in zip(df.to_dict(orient="records"), vectors):
        data_obj = {
            "name": row["name"],
            "type": row["type"],
            "city": row["city"],
            "description": row["description"],
            "contact": row["contact_info"],
        }
        client.data_object.create(data_obj, class_name=CLASS_NAME, vector=vec.tolist())
    print("✅ Upload complete.")
else:
    print(f"ℹ️  {count} objects already in Weaviate – skipping upload.")

# ------------------------------------------------------------------
# 3  Search helper
# ------------------------------------------------------------------
def semantic_search(query: str, k: int = 3):
    q_vec = embedder.encode(query).tolist()
    res = (
        client.query.get(CLASS_NAME, ["name", "city", "description", "contact"])
              .with_near_vector({"vector": q_vec})
              .with_limit(k)
              .do()
    )
    return res["data"]["Get"][CLASS_NAME]

# ------------------------------------------------------------------
# 4  CLI loop
# ------------------------------------------------------------------
print("\n🧠 Oklahoma Resource Agent – Free LLM ✦ type ‘exit’ to quit\n")
while True:
    q = input("What do you need help with?\n> ").strip()
    if q.lower() in {"exit", "quit"}:  break
    if not q:                            continue
    matches = semantic_search(q)
    if not matches:
        print("No matching resources – try again.\n"); continue

    print("\n🔎 Top Matches:\n")
    for i, m in enumerate(matches, 1):
        print(f"{i}. {m['name']} ({m['city']})\n   {m['description']}\n   ➤ Contact: {m['contact']}\n")

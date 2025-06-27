from flask import Flask, request, jsonify
import chromadb
from fastembed import TextEmbedding
import requests, os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Configuration
DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
COLLECTION = os.getenv("CHROMA_COLLECTION", "leave_policy_pdfs")
EMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-base-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_MAX = int(os.getenv("LLM_MAX_TOKENS", "500"))
API_URL = os.getenv("CUSTOM_API_ENDPOINT")
API_KEY = os.getenv("CUSTOM_API_KEY")

# Initialize ChromaDB and embedding model
base = os.path.dirname(os.path.abspath(__file__))
chroma = chromadb.PersistentClient(path=os.path.join(base, DB_DIR))
embedder = TextEmbedding(model_name=EMBED_MODEL)

def embed_text(text):
    return next(embedder.embed([text]))

def query_db(vec, k=5):
    col = chroma.get_or_create_collection(name=COLLECTION)
    return col.query(
        query_embeddings=[vec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

def clean_text(s: str) -> str:
    # Remove all newline and carriage return characters
    # Also collapse multiple spaces
    text = s.replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())

def gen_response(q: str, top: dict) -> str:
    docs = top["documents"][0]
    metas = top["metadatas"][0]
    if not docs:
        return "No indexed documents available. Please ingest PDF first."

    # Clean up doc chunks
    clean_docs = [clean_text(d) for d in docs]
    context = "\n\n".join(
        f"Document: {d}\nMetadata: {m}"
        for d, m in zip(clean_docs, metas)
    )

    prompt = (
        "You are an assistant on leave policy.\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {q}\nAnswer:"
    )

    resp = requests.post(
        API_URL,
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": LLM_MAX,
            "temperature": 0.3
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30
    )

    if resp.ok:
        return resp.json()["choices"][0]["message"]["content"].strip()
    return f"API error: {resp.status_code}"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    q = data.get("question")
    if not q:
        return jsonify({"error": "Provide 'question'"}), 400

    vec = embed_text(q)
    top = query_db(vec)
    answer = gen_response(q, top)

    # Clean top_results documents for cleaner response
    top["documents"][0] = [clean_text(d) for d in top["documents"][0]]
    return jsonify({"answer": answer, "top_results": top})

@app.route("/health", methods=["GET"])
def health():
    try:
        col = chroma.get_or_create_collection(COLLECTION)
        return jsonify({"status": "healthy", "count": col.count()})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == "__main__":
    print("⚙️ Flask running with ChromaDB at", os.path.join(base, DB_DIR))
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "").lower() == "true"
    )

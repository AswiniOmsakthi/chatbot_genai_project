#!/usr/bin/env python3
import sys, os
import chromadb
from fastembed import TextEmbedding
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()
client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_DIR"))
embedder = TextEmbedding(model_name=os.getenv("FASTEMBED_MODEL"))

COLLS = {
    "leave": os.getenv("CHROMA_LEAVE"),
    "travel": os.getenv("CHROMA_TRAVEL"),
    "harass": os.getenv("CHROMA_HARASS")
}

def extract_chunks(pdf_path, chunk_size=500):
    reader = PdfReader(pdf_path)
    chunks = []
    for p, pg in enumerate(reader.pages):
        text = pg.extract_text() or ""
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append((chunk, {"source": os.path.basename(pdf_path), "page": p+1}))
    return chunks

def main():
    if len(sys.argv) < 3 or sys.argv[1] not in COLLS:
        sys.exit("Usage: ingest_doc.py [leave|travel|harass] <file1.pdf> ...")
    domain = sys.argv[1]
    coll = COLLS[domain]
    try:
        client.delete_collection(coll)
    except:
        pass
    collection = client.create_collection(name=coll)
    for pdf in sys.argv[2:]:
        if not os.path.isfile(pdf):
            print(f"❌ File not found: {pdf}")
            continue
        for text, meta in extract_chunks(pdf):
            emb = next(embedder.embed([text]))
            collection.add(
                embeddings=[emb],
                documents=[text],
                metadatas=[meta],
                ids=[f"{meta['source']}_p{meta['page']}"]
            )
    print(f"[{domain}] ingested {collection.count()} chunks")

if __name__ == "__main__":
    main()

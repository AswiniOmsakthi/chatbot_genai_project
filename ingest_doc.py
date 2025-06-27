#!/usr/bin/env python3
import sys, os
import chromadb
from fastembed import TextEmbedding
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

def extract_pdf_chunks(pdf_path, chunk_size=500):
    chunks = []
    reader = PdfReader(pdf_path)
    print(f"📄 Processing {len(reader.pages)} pages...")
    for p, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append((chunk, {
                    "page": p + 1,
                    "source": os.path.basename(pdf_path),
                    "chunk_id": i // chunk_size
                }))
    print(f"✅ Extracted {len(chunks)} chunks")
    return chunks

def main(pdf_path):
    if not os.path.isfile(pdf_path):
        print("❌ PDF not found:", pdf_path)
        sys.exit(1)

    base = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.getenv("CHROMA_DB_DIR", "chroma_db")
    collection = os.getenv("CHROMA_COLLECTION", "leave_policy_pdfs")
    chroma = chromadb.PersistentClient(path=os.path.join(base, db_dir))

    try:
        chroma.delete_collection(collection)
        print("🗑️  Cleared old collection")
    except:
        pass

    col = chroma.create_collection(name=collection, metadata={"description": "Leave policy docs"})
    chunks = extract_pdf_chunks(pdf_path)
    if not chunks:
        sys.exit("❌ No chunks — exiting")

    texts, metas = zip(*chunks)
    embedder = TextEmbedding(model_name=os.getenv("FASTEMBED_MODEL"))
    embeddings = list(embedder.embed(texts))
    ids = [f"{os.path.basename(pdf_path)}_p{m['page']}_c{m['chunk_id']}" for m in metas]

    col.add(embeddings=embeddings, documents=list(texts), metadatas=list(metas), ids=ids)
    print("💾 Ingested:", col.count(), "chunks")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python ingest_doc.py <pdf_path>")
    main(sys.argv[1])

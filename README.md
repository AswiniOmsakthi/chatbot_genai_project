# Leave Policy QA API

This project provides a Flask API for answering questions about your company's leave policy using PDF documents. It uses FastEmbed for embedding, ChromaDB as a vector database, and Psitron Tech's LLM API for generating meaningful answers.

## Features
- Upload and embed leave policy PDFs (metadata stored in ChromaDB)
- Query API with natural language questions
- Returns top 5 relevant policy sections and a concise answer from Psitron Tech LLM
- Environment-based configuration using .env file
- Enhanced reasoning capabilities with custom LLM endpoint

## Quick Start

### 1. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
```

**Linux/macOS:**
```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Copy and Edit Environment Variables
```bash
# Copy the example environment file
cp env_example.txt .env
# Edit the .env file with your settings
```

### 5. Ingest Policy Documents
```bash
python -m src.ingest_doc leave data/leave_policy.pdf
python -m src.ingest_doc travel data/travel_policy.pdf
python -m src.ingest_doc harass data/harassment_policy.pdf
```

### 6. Start the API Server
```bash
python -m src.app
```
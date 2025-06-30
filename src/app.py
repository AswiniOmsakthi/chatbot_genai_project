from flask import Flask, request, jsonify
import chromadb, os
from fastembed import TextEmbedding
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType

load_dotenv()
app = Flask(__name__)

# Initialize databases and embedder
client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_DIR"))
embedder = TextEmbedding(model_name=os.getenv("FASTEMBED_MODEL"))

# Initialize Azure GPT-4
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("LLM_MODEL"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0.2
)

COLLS = {
    "leave": os.getenv("CHROMA_LEAVE"),
    "travel": os.getenv("CHROMA_TRAVEL"),
    "harass": os.getenv("CHROMA_HARASS")
}

def clean_text(s: str) -> str:
    return " ".join(s.replace("\n", " ").split())

def make_tool(domain: str, coll_name: str) -> Tool:
    def query_fn(query: str) -> str:
        vec = next(embedder.embed([query]))
        res = client.get_or_create_collection(coll_name).query(
            query_embeddings=[vec],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        formatted = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            formatted.append(f"{clean_text(doc)} (page {meta.get('page')}, dist {dist:.3f})")
        return "\n\n".join(formatted) or "No relevant content found."
    return Tool(
        name=f"{domain}_policy",
        func=query_fn,
        description=f"Search the {domain} policy documents."
    )

tools = [make_tool(d, c) for d, c in COLLS.items()]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3
    
)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question")
    if not question:
        return jsonify({"error": "Provide 'question'"}), 400
    try:
        answer = agent.run(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "False").lower() == "true"
    )




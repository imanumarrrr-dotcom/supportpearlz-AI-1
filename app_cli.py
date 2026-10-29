"""Optional CLI interface; requires OPENAI_API_KEY in the environment."""
import os
from src.config import settings
from src.retrieval.vector_store import load_vector_store
from src.chains.rag_chain import answer_question

def main():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY to use the CLI, or run Streamlit to enter a key securely.")
    store = load_vector_store(key, settings.vector_store_path, settings.embedding_model)
    history = []
    print("SupportPearlz CLI. Type /reset to clear history or /quit to exit.")
    while True:
        q = input("You > ").strip()
        if q == "/quit": break
        if q == "/reset": history = []; print("History cleared."); continue
        if not q: continue
        result = answer_question(q, history, store, key, settings.model, settings.temperature, settings.retrieval_k, settings.score_threshold)
        print(f"Bot > {result['answer']}\nSources: {result['sources']} | confidence={result['confidence']}")
        history.extend([{"role":"user","content":q},{"role":"assistant","content":result["answer"]}])
if __name__ == "__main__":
    main()

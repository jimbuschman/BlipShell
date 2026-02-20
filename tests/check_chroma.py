"""Quick check: what does ChromaDB return for a test query?"""
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

ef = OllamaEmbeddingFunction(
    url="http://localhost:11434", model_name="nomic-embed-text"
)
c = chromadb.PersistentClient(path="data/chroma")
coll = c.get_collection("memories", embedding_function=ef)

r = coll.query(
    query_texts=["python performance"],
    n_results=5,
    include=["distances", "documents"],
)

for i in range(len(r["ids"][0])):
    d = r["distances"][0][i]
    doc = r["documents"][0][i][:100]
    print(f"sim={1-d:.3f}  dist={d:.3f}  id={r['ids'][0][i]}  {doc}")

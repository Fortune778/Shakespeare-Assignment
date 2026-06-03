import chromadb
import ollama

from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "shakespeare"

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
LLM_MODEL = "tinyllama"

TOP_K = 3

# =========================
# CONNECT TO CHROMADB
# =========================

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_collection(COLLECTION_NAME)

print("RAG system ready!")

# =========================
# MAIN LOOP
# =========================

while True:

    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    # =========================
    # EMBED QUESTION
    # =========================

    # embedding_response = ollama.embeddings(
    #     model=EMBED_MODEL,
    #     prompt=question
    # )

    # question_embedding = embedding_response["embedding"]
    question_embedding = EMBED_MODEL.encode(question).tolist()
    # =========================
    # VECTOR SEARCH
    # =========================

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )

    retrieved_docs = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    # =========================
    # SHOW RETRIEVED CONTEXT
    # =========================

    print("\n" + "=" * 50)
    print("RETRIEVED CONTEXT")
    print("=" * 50)

    context_parts = []

    for i, (doc, meta) in enumerate(
        zip(retrieved_docs, retrieved_metadata),
        start=1
    ):

        print(f"\nChunk {i}")
        print("-" * 50)

        print(
            f"{meta.get('play')} | "
            f"Act {meta.get('act')} | "
            f"Scene {meta.get('scene')}"
        )

        print("\nTEXT:\n")
        print(doc[:700])

        context_parts.append(doc)

    # =========================
    # BUILD PROMPT
    # =========================

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a Shakespeare literature assistant.

Answer ONLY using the provided context.

If the answer is not in the context, say:
"I could not find enough evidence in the text."

Context:
{context}

Question:
{question}

Answer:
"""

    # =========================
    # GENERATE RESPONSE
    # =========================

    print("\n" + "=" * 50)
    print("AI ANSWER")
    print("=" * 50 + "\n")

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True,
        options={
            "temperature": 0.3
        }
    )

    for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)

    print("\n")
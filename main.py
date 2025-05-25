
import os
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from ollama import Client as OllamaClient
from serpapi import GoogleSearch
from serpapi.exceptions import HTTPError

# === Configuration ===
CHROMA_DIR = "./kcc_chroma"
COLLECTION_NAME = "kcc_qna"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma:7b"
RELEVANCE_THRESHOLD = 0.75
TOP_K = 5
SERPAPI_API_KEY = "ea6786617d5200739025a4cec1f15116664957b0e34e7d61c23ad959f09232cd"

# === Initialize components ===
embedder = SentenceTransformer(EMBED_MODEL)
llm_client = OllamaClient(host="http://localhost:11434")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

# === Helper functions ===

def query_llm(prompt: str) -> str:
    response = llm_client.generate(model=OLLAMA_MODEL, prompt=prompt, stream=False)
    return response["response"]

def semantic_search(query: str):
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    return results

def generate_prompt(query, results):
    if not results or not results["documents"]:
        return None, None
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join([f"Q: {meta['question']}\nA: {doc}" for doc, meta in zip(chunks, metadatas)])
    prompt = f"""You are an expert answering questions based on the following KCC data:

{context}

Now answer the question: \"{query}\""""
    return prompt, context

def is_context_sufficient(query: str, context: str) -> bool:
    check_prompt = f"""You are a helpful assistant. Based on the context below, decide whether you have enough information to answer the question so that I can reiterate for the answer.

Context:
{context}

Question:
{query}

Answer with only "YES" or "NO".
"""
    response = query_llm(check_prompt)
    return response.strip().upper() == "YES"

def search_google(query: str) -> str:
    if not SERPAPI_API_KEY:
        return "❌ SerpAPI key not set. Cannot perform web search."

    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 3,
            "engine": "google",
        })

        results = search.get_dict()
        organic = results.get("organic_results", [])
        if not organic:
            return "⚠️ No results found from Google search."

        return "\n\n".join(
            f"🔗 [{item['title']}]({item['link']})\n\n{item.get('snippet', '')}" for item in organic
        )
    except HTTPError as e:
        return f"❌ Failed to search Google: {e}"

def answer_question(query: str) -> dict:
    results = semantic_search(query)

    if results["documents"] and any(dist < (1 - RELEVANCE_THRESHOLD) for dist in results["distances"][0]):
        prompt, context = generate_prompt(query, results)
        if prompt and is_context_sufficient(query, context):
            response = query_llm(prompt)
            return {
                "source": "KCC",
                "answer": response.strip(),
                "context": context
            }

    # Fallback to web search
    web_result = search_google(query)
    web_prompt = f"""You could not find the answer in the KCC database. Based on the following web search results, answer the question:

{web_result}

Now answer the question: \"{query}\"

Note: You are an expert in agriculture and rural development. You are also an expert in the KCC services and policies. You need to answer based on agriculture and rural development context only."""

    final_answer = query_llm(web_prompt)
    return {
        "source": "Internet (Google)",
        "answer": final_answer.strip(),
        "context": web_result
    }

# === Streamlit UI ===

st.set_page_config(page_title="KCC Chat Assistant", page_icon="🌾")

st.title("🧑‍🌾 KCC Query Assistant")
st.caption("Ask your agricultural or rural development questions.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "answered_questions" not in st.session_state:
    st.session_state.answered_questions = {}
 
# Display chat history
if not st.session_state.chat_history:
    with st.chat_message("assistant"):
        st.markdown("👋 Hello! How can I assist you today? Ask me anything related to agriculture or rural services.")
else:
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])

        with st.chat_message("assistant"):
            st.markdown(f"**Answer (from {chat['answer']['source']}):**\n\n{chat['answer']['answer']}", unsafe_allow_html=True)

            if chat["answer"]["source"] == "KCC":
                with st.expander("📄 Retrieved KCC Context"):
                    st.code(chat["answer"]["context"], language="markdown")
            else:
                with st.expander("🌐 Web Context"):
                    st.markdown(chat["answer"]["context"], unsafe_allow_html=True)

if query:=st.chat_input("Ask your question"):
    if query in st.session_state.answered_questions:
        result = st.session_state.answered_questions[query]
    else:
        with st.chat_message("user"):
            st.markdown(query)
        with st.spinner("Thinking..."):
            result = answer_question(query)
        st.session_state.answered_questions[query] = result

    st.session_state.chat_history.append({"question": query, "answer": result})
    with st.chat_message("assistant"):
        st.markdown(f"**Answer (from {result['source']}):**\n\n{result['answer']}", unsafe_allow_html=True)

        if result["source"] == "KCC":
            with st.expander("📄 Retrieved KCC Context"):
                st.code(result["context"], language="markdown")
        else:
            with st.expander("🌐 Web Context"):
                st.markdown(result["context"], unsafe_allow_html=True)

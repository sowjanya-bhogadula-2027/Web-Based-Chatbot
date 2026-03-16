import streamlit as st
import time
import os
import bs4
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain

load_dotenv()

# --- Configuration & Initialization ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🌐 Web Content RAG Chat")

# Initialize Session State
if "store" not in st.session_state:
    st.session_state.store = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

# --- Session Timeout Logic (2 Minutes) ---
current_time = time.time()
if current_time - st.session_state.last_activity > 120:
    st.session_state.chat_history = []
    st.session_state.store = {}
    st.warning("Session expired due to 2 minutes of inactivity. History cleared.")

st.session_state.last_activity = current_time

# --- Sidebar: URL Input & Processing ---
with st.sidebar:
    url_input = st.text_input("Enter Web URL:", placeholder="https://example.com")
    process_button = st.button("Process Content")

@st.cache_resource # Cache the vectorstore to avoid re-loading on every message
def get_vectorstore(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=splits, embedding=embeddings)

# --- RAG Logic Setup ---
if url_input and process_button:
    with st.spinner("Processing URL..."):
        st.session_state.vectorstore = get_vectorstore(url_input)
        st.success("Content Loaded!")

# --- Chat Interface ---
if "vectorstore" in st.session_state:
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
    retriever = st.session_state.vectorstore.as_retriever()

    # Contextualize Question Prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA Prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say "
        "that you don't know. Use three sentences maximum.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask something about the content..."):
        st.session_state.chat_history.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            # Convert session history to LangChain format for the chain
            formatted_history = []
            for m in st.session_state.chat_history[:-1]:
                if m["role"] == "human":
                    formatted_history.append(("human", m["content"]))
                else:
                    formatted_history.append(("ai", m["content"]))

            response = rag_chain.invoke({"input": prompt, "chat_history": formatted_history})
            full_response = response["answer"]
            st.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
else:
    st.info("Please enter a URL in the sidebar and click 'Process Content' to start.")
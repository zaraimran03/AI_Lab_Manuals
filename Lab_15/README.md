# Lab 15: Build a RAG-Based Chatbot Using Ollama, LangChain, and Streamlit

## Lab Overview
Complete RAG chatbot implementation using local LLM (Ollama), LangChain framework, and Streamlit UI.

## Objectives
- Set up Ollama locally with LLAMA2
- Build LangChain chatbot
- Implement document ingestion and vector storage
- Enable RAG functionality
- Deploy with Streamlit

## Tasks Completed

### Step 1-2: Ollama Setup
- Installed Ollama
- Downloaded LLAMA2 model
- Verified installation with `ollama run llama2`

### Step 3-5: Basic Chatbot
- Created project structure
- Installed dependencies (langchain, streamlit, ollama)
- Built basic prompt template
- Implemented Streamlit UI

### Step 6-8: API Integration
- Set up environment variables (.env)
- Configured LangChain API key
- Enabled tracing
- Tested OpenAI integration

### Step 9-13: RAG Implementation
- Loaded documents (TextLoader)
- Split into chunks (RecursiveCharacterTextSplitter)
- Created embeddings (OllamaEmbeddings)
- Built FAISS vector store
- Created retriever
- Modified prompt for RAG context
- Built RAG chain

### Step 14: Deployment
- Ran application with `streamlit run app.py`
- Tested question-answering
- Verified context-based responses

## Files Created
1. `app.py` - Streamlit application
2. `localama.py` - Ollama integration
3. `.env` - Environment variables
4. `requirements.txt` - Dependencies
5. `data/sample_docs.txt` - Sample documents

## Libraries Used
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
```

## Key Learnings
- RAG enhances LLM responses with external knowledge
- Document chunking important for retrieval
- Vector embeddings enable semantic search
- FAISS provides efficient similarity search
- Streamlit simplifies UI development
- Environment management best practices
- Local LLMs (Ollama) vs cloud APIs
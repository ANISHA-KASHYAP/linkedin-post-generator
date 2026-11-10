# LinkedIn Post Generator (RAG)

A GenAI app that writes LinkedIn posts in your own style using Llama.

## Features
- Tone, length and audience controls
- Few-shot prompting
- RAG: learns your style from your past posts (ChromaDB)
- Multiple variations, "Improve my draft" mode
- Copy and download

## Tech
Python, Streamlit, Llama 3 (Ollama locally, Groq online), ChromaDB

## Run locally
1. Install Ollama and run: ollama pull llama3.2:3b
2. pip install -r requirements.txt
3. python -m streamlit run app.py
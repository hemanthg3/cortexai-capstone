---
title: CortexAI Multi-Agent Research Assistant
emoji:   
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# CortexAI - Multi-Agent Research Assistant

## Capstone Project

This project is a Day 10 capstone build that integrates:

- Retrieval-Augmented Generation
- Multi-agent workflow
- Streamlit UI
- Structured JSON outputs
- Logging
- Evaluation harness
- Docker deployment
- Hugging Face Spaces deployment readiness

## Features

- Paste company data, dataset text, article text, or notes
- Ask a research question
- Retrieve relevant context using vector search
- Run a multi-agent workflow:
  - Planner Agent
  - Researcher Agent
  - Critic Agent
  - Formatter Agent
- Generate validated structured JSON output
- Download results as JSON
- Run basic evaluation checks

## Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- Groq
- ChromaDB
- HuggingFace Embeddings
- Pydantic
- Docker

## Environment Variables

Create a secret/environment variable:

```env
GROQ_API_KEY=your_groq_api_key_here
```
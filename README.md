# RAG Workshop: Building a PDF-Based Chatbot for Persian Documents

This repository contains the materials, notebooks, and code used in a hands-on workshop I presented titled "Development of an Offline Persian AI Assistant using RAG" at the WTLAB 2025 workshops. This workshop focused on building a **Retrieval-Augmented Generation (RAG)** chatbot for **Persian PDF documents**, such as legal regulations and organizational rules.

The workshop demonstrates how to extract text from PDFs, chunk and index the content, retrieve relevant documents, and generate final answers using open-source **Large Language Models (LLMs)**. A simple **Streamlit** web interface is also included for interactive chat.

---

## Workshop Goals

* Understand the basics of **LLMs** and **Retrieval-Augmented Generation (RAG)**
* Learn how to choose and use open-source models
* Learn how to **parse Persian PDF documents** efficiently
* Convert text to structured and searchable representations
* Implement a **RAG pipeline** to answer user questions from private domain documents with citations
* Deploy a simple **chatbot UI** using Streamlit

---

## Repository Structure

| File / Folder                                             | Description                                           |
| --------------------------------------------------------- | ----------------------------------------------------- |
| `0.About-me.ipynb`                                        | Workshop introduction and presenter info              |
| `1.Introduction-to-LLMs-And-RAG.ipynb`                    | Overview of LLMs and RAG concepts                     |
| `2.GettingStarted-with-Open-Source-LLMs.ipynb`            | Practical intro to running and choosing open-source models         |
| `3.RAG-Workflow.ipynb`                                    | Step-by-step explanation of RAG architecture          |
| `4.Parsing-PDFs.ipynb`                                    | Extracting and cleaning text from Persian PDFs        |
| `5.Implement_RAG.ipynb`                                   | Building a RAG system from scratch                    |
| `8.More-Advanced-RAG.ipynb`                               | Experimenting with more advanced methods, Parent-Child Retriever, GraphRAG, etc. |
| `7.streamlit_ui.py`                                       | Minimal chatbot UI built with Streamlit               |
| `data-sources/`                                           | Sample PDF regulation document used in the workshop  |
| `docling-pdf-parse/`                                       | Parsing PDF using docling
| `pyproject.toml`, `uv.lock`                               | Environment configuration for reproducibility         |

---

## Running the RAG Chatbot

### 1) Prepare the vector index (optional depending on notebook steps)

Run the relevant notebook to generate chunk embeddings.

### 2) Launch the Streamlit UI:

```bash
streamlit run 7.streamlit_ui.py
```

This opens a web interface where you can interact with your chatbot.

---

## Notes on Persian PDF Processing

Persian PDFs are often encrypted, scanned, or improperly encoded. In this workshop, we compare multiple parsing approaches:

* **PyPDF2**
* **OCR-based extraction**
* **Docling structured document parsing**

Pros/cons and preprocessing techniques are discussed in the notebooks.

---

## Future Extensions

You may extend this project by:

* Using **FAISS** / **Qdrant** / **Weaviate** for scalable vector search
* Implementing **GraphRAG** (entity/relation extraction → knowledge graphs) or **Parent-Child Document Retriever**

---

## Author

**Ali Moameri**
Software Engineer & NLP Researcher
Master's Student at Ferdowsi University of Mashhad

---

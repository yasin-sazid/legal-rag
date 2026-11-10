# Legal Document Retrieval-Augmented Generation

A Python project for conversational question answering over a local collection of legal PDFs. It combines document ingestion, embedding-based retrieval, Chroma vector storage, and an Ollama-hosted language model, with a second workflow focused on multilingual retrieval and relevance-guided query refinement.

## Two retrieval workflows

| Script | Approach |
| --- | --- |
| `rag_script.py` | MiniLM embeddings, top-three retrieval, and Llama 3 answer generation |
| `bangla_legal_rag.py` | Multilingual BGE-M3 embeddings, top-five candidates, LLM relevance assessment, and iterative query refinement |

The multilingual script selects `BAAI/bge-m3` with a MiniLM fallback. It uses Llama 3 both for relevance assessment and final response generation, permits up to three refinement iterations, and synthesizes an answer from the selected context with the `refine` response mode.

## Architecture

```text
pdfs/ → SimpleDirectoryReader → Embeddings → Chroma-backed index
                                                 |
Question → Retrieval → [Relevance / query refinement] → Ollama → Answer
```

Both scripts build their index from the documents loaded at startup and use `chromadb.Client()` for the vector-store client. The interactive loop accepts questions until the user enters `exit`.

## Setup and usage

Use Python with the dependencies recorded in `requirements.txt`, and run an Ollama service with `llama3:latest` available. The dependency file is UTF-16 encoded; pip can read the BOM-marked file directly.

```bash
python -m pip install -r requirements.txt
ollama pull llama3:latest
```

Supply the PDF corpus in a `pdfs/` directory beneath the working directory, then choose a workflow:

```bash
python rag_script.py
# or
python bangla_legal_rag.py
```

Embedding models are loaded through Hugging Face integrations. LlamaIndex connects document ingestion, embedding adapters, retrieval, and response synthesis, while LangChain Community supplies the embedding wrapper.

## Technical contributions

The project demonstrates local-model RAG orchestration and a multilingual retrieval workflow that makes relevance decisions, reformulated queries, and retrieved-context counts visible during interactive use.

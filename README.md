# Texas RRC Waste Management — RAG Knowledge Base Chatbot
 
A Retrieval Augmented Generation (RAG) chatbot that answers questions about Texas Railroad Commission (RRC) waste management regulations. Built with LlamaIndex, Mistral AI, Pinecone, and Streamlit.
 
## Overview
 
Regulatory documents from the Texas RRC are crawled, chunked, embedded, and stored in a Pinecone vector index. A Streamlit chat interface lets users ask questions and receive grounded answers with source citations, powered by Mistral NeMo as the LLM and Mistral Embed for vector search.

### Architecture
 
```
Texas RRC website
        │
        ▼
  Web Crawler (Crawl4AI)
        │
        ▼
  Markdown docs (crawled-docs/)
        │
        ▼
  LlamaIndex Ingestion Pipeline
    ├── SentenceSplitter (512 tokens, 50 overlap)
    └── MistralAI Embedding (mistral-embed)
        │
        ▼
  Pinecone Vector Store ("texas-wastemanagment")
        │
        ▼
  Streamlit Chat UI ◄── Mistral NeMo LLM (open-mistral-nemo)
    ├── ChatMemoryBuffer
    ├── DuplicateRemoverPostProcessor 
    └── SentenceEmbeddingOptimizer 
```

## Key Features
 
- **Domain specific RAG** — grounded in actual Texas RRC regulatory text, not generic LLM knowledge
- **Custom post-processing** — `DuplicateRemoverPostProcessor` uses Jaccard similarity to deduplicate retrieved chunks before they reach the LLM
- **Retrieval optimization** — `SentenceEmbeddingOptimizer` applies percentile and threshold cutoffs to filter low relevance sentences from retrieved nodes
- **Conversational memory** — `ChatMemoryBuffer` maintains multi-turn context within a session
- **Source transparency** — every answer displays scored source nodes with file names so users can verify claims against the original documents
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| LLM | Mistral NeMo (`open-mistral-nemo`) |
| Embeddings | Mistral Embed (`mistral-embed`) |
| Orchestration | LlamaIndex |
| Vector Store | Pinecone |
| Web Crawling | Crawl4AI |
| Frontend | Streamlit |
| Package Manager | uv |

```
├── main.py                        # Streamlit chat application
├── ingestion.py                   # Pinecone ingestion pipeline
├── crawl_docs.py                  # Text/Markdown document crawler
├── crawled-docs/                  # Downloaded RRC regulatory documents
├── pyproject.toml                 # Project metadata and dependencies (uv)
├── uv.lock                        # Lockfile
└── .gitignore
```
## How It Works
 
1. **Crawl** — `crawl_docs.py` scrapes Texas RRC waste management pages and saves them as Markdown files.
2. **Ingest** — `ingestion.py` reads the Markdown files, splits them into 512 token chunks with 50-token overlap, embeds each chunk with `mistral-embed`, and upserts vectors to Pinecone.
3. **Query** — When a user asks a question in the Streamlit UI, LlamaIndex retrieves the top 3 most relevant chunks from Pinecone, deduplicates and filters them through the post processors, then sends the refined context plus the question to Mistral NeMo for a grounded answer.

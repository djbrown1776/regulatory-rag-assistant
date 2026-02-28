from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor, SummaryExtractor, KeywordExtractor
import asyncio
from llama_index.core.storage.docstore import SimpleDocumentStore
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
#from dotenv import load_dotenv

llm = Ollama(model="ministral-3:14b", request_timeout=120.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text-v2-moe:latest")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 250  #512
Settings.chunk_overlap = 25  #50

persistence_dir = "./pipeline_storage"
chroma_dir = "./chroma.db"

def get_transformations():
    return [
        SentenceSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap
            ),
        TitleExtractor(),
        #SummaryExtractor(),
        #KeywordExtractor(),
        Settings.embed_model
    ]

async def main():
    print("Loading Documents...")
    documents = SimpleDirectoryReader(
        input_dir="./llamaindex-docs",
        required_exts=[".md"],
        num_files_limit=5  #10
    ).load_data()
    print(f"Loaded {len(documents)} documents.")

    #Create persistent Chroma vector store
    print("Setting uo ChromaDB")
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    chroma_collection = chroma_client.get_or_create_collection("llamaindex_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    #Count of docs in vector store
    existing_count = chroma_collection.count()
    print(f"ChromaDB already contains {existing_count} embeddings.")

    #if exsist in embeding skip ingest and query
    if existing_count > 0:
        print("Using existing embeddings from ChromaDB")
    else:
    #Create Pipeline
        pipeline = IngestionPipeline(
            transformations=get_transformations(),
           vector_store=vector_store,
        )

        #Run Pipeline
        processed_nodes = await pipeline.arun(
            documents=documents,
            show_progress=True,
            num_workers=1
        )
        print(f"Processed {len(processed_nodes)} nodes with ChromaDB.")

    #Verify Embeddings
        if processed_nodes[0].embedding:
            print(f"Embedding dimesnions: {len(processed_nodes[0].embedding)}")

        #Display metadata if available
        #first_node_metadata = processed_nodes[0].metadata
        #print("First node metedate:")
        #for key, value in first_node_metadata.items():
        #    print(f" {key}: {value}")

        #Create Vector Store Index from nodes
    print("Create Vector Store Index for ChromaDB...")
    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    print("Vector Store Index created succesfully.")

        #Creat query Engine
    query_engine = vector_index.as_query_engine()

    response = query_engine.query("What is the IngestionPipeline?")
    print("Simple Query Response:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())

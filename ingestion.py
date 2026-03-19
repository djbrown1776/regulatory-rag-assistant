import os
import time
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
# Switched to Mistral AI imports
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "texas-wastemanagment"
# Use your Mistral API Key from your .env
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Set up Mistral NeMo
llm = MistralAI(model="open-mistral-nemo", api_key=MISTRAL_API_KEY)

# Mistral's dedicated embedding model
embed_model = MistralAIEmbedding(
    model_name="mistral-embed", 
    api_key=MISTRAL_API_KEY,
    embed_batch_size=50  # Mistral's optimal batch size
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 50

def main():
    print("=" * 60)
    print("Texas RRC Waste Management Ingestion (Mistral NeMo Edition)")
    print("=" * 60)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    print("Loading RRC documents...")
    documents = SimpleDirectoryReader(
        input_dir="./crawled-docs",
        required_exts=[".md"],
    ).load_data()
    print(f"  Loaded {len(documents)} documents")

    print("Creating ingestion pipeline...")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap,
            ),
            embed_model,
        ],
        vector_store=vector_store,
    )

    print("\n[4/5] Running ingestion...")
    start_time = time.time()
    nodes = pipeline.run(documents=documents, show_progress=True)
    elapsed = time.time() - start_time
    print(f"\n  ✅ Ingestion complete in {elapsed:.2f}s — {len(nodes)} nodes")

    print("\n[5/5] Testing query...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine()
    response = query_engine.query("What are the main requirements of Statewide Rule 8?")
    print(f"\n  Query response: {response}")

if __name__ == "__main__":
    main()
from litellm.types.rerank import Required
from llama_index.core import SimpleDirectoryReader, Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.semantic_splitter import SentenceCombination
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

llm = Ollama(model="gemma3:latest", request_timeout=120.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

def main():
    documents = SimpleDirectoryReader(
        input_dir="./llamaindex-docs",
        recursive=False,
        required_exts=[".md"],
        num_files_limit=20,
    ).load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=SentenceSplitter(),
    )

    query_engine = index.as_query_engine()
    response = query_engine.query("How to integrate pinecone as the vector databse?")
    print(response)

if __name__ == "__main__":
    main()

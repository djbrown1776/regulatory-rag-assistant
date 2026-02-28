from litellm.types.rerank import Required
from llama_index.core import SimpleDirectoryReader, Document, Settings, VectorStoreIndex
from llama_index.core import node_parser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.semantic_splitter import SentenceCombination
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

llm = Ollama(model="gemma3:latest", request_timeout=120.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 50

def main():
    documents = SimpleDirectoryReader(
        input_dir="./llamaindex-docs",
        required_exts=[".md"],
        num_files_limit=10
    ).load_data()

    print(f"Loaded {len(documents)} documents.")

    node_parser = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
    )

    print("Parsing documents into nodes with custom chunking...")
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"Parsed {len(nodes)} nodes from documents.")

    #Inspect a few sample nodes
    print("\nSample nodes after custom chunking:")
    for i, node in enumerate(nodes[:3]):
        print(f"\nNode {i+1} content:\n{node.get_content()}\n")

        #Display metadata if available
        if node.metadata:
            print(f"- source: {node.metadata.get('file_name', 'N/A')}")

    #Create Vector Store Index from nodes
    print("Create Vector Store Index nodes...")
    index = VectorStoreIndex(nodes)
    print("Vector Store Index created succesfully.")

    #Example query to test the index
    query = "How does Llamaindex work with Amazon Cloud?"
    print(f"\nQuerying the index with: '{query}'")
    response = index.as_query_engine().query(query)
    print(f"Response:\n{response}")

if __name__ == "__main__":
    main()

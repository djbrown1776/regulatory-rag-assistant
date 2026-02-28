import os

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.web import SimpleWebPageReader


def main():
    llm = Ollama(model="gemma3:latest", request_timeout=120.0)
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    Settings.llm = llm
    Settings.embed_model = embed_model

    url = "https://meidasnews.com/news/new-report-highlights-ice-and-cbps-multi-million-dollar-surge-to-amass-weapons-and-munitions-executive"
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    query = "Summarize this article hitting the key points?"
    response = query_engine.query(query)
    print(f"Response:\n {response}\n")


if __name__ == "__main__":
    main()

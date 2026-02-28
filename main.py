import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
import streamlit as st
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

load_dotenv()

# Configuration
INDEX_NAME = "texas-wastemanagment"
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


class DuplicateRemoverPostProcessor(BaseNodePostprocessor):
    """Post-processor to remove duplicate nodes based on text content."""

    similarity_threshold: float = 0.8  # Jaccard similarity threshold

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle = None
    ) -> list[NodeWithScore]:

        if not nodes:
            return nodes

        seen_texts = []
        unique_nodes = []

        for node in nodes:
            node_text = node.node.get_content()
            is_duplicate = False

            for seen_text in seen_texts:
                node_words = set(node_text.lower().split())
                seen_words = set(seen_text.lower().split())

                overlap = len(node_words & seen_words)
                total = len(node_words | seen_words)
                jaccard_similarity = overlap / total if total > 0 else 0

                if jaccard_similarity > self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_nodes.append(node)
                seen_texts.append(node_text)

        return unique_nodes


@st.cache_resource
def get_index():
    """Connect to Pinecone vector store and return index."""
    print("Connecting to Pinecone vector store...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def main():
    st.set_page_config(
        page_title="Texas RRC Waste Management Assistant", layout="wide", page_icon="📚"
    )
    st.title("Texas RRC Waste Management Assistant")
    st.caption("Ask questions about Texas RRC waste management regulations.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_engine" not in st.session_state:
        index = get_index()
        memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

        sentence_optimizer = SentenceEmbeddingOptimizer(
            embed_model=Settings.embed_model,
            percentile_cutoff=0.5,
            threshold_cutoff=0.7,
            context_before=1,
            context_after=1,
        )

        st.session_state.chat_engine = index.as_chat_engine(
            memory=memory,
            chat_mode=ChatMode.BEST,
            similarity_top_k=3,
            post_processors=[DuplicateRemoverPostProcessor(), sentence_optimizer],
            system_prompt=(
                "You are a helpful assistant that answers questions about Texas RRC "
                "waste management regulations. Use the retrieved context to provide "
                "accurate, helpful answers. If you don't know the answer, say so."
            ),
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about Texas RRC waste management..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)

            st.markdown(response.response)

            nodes = [node for node in response.source_nodes]
            if nodes:
                with st.expander(f"📚 Sources ({len(nodes)} documents)"):
                    for i, node in enumerate(nodes, 1):
                        score = node.score if node.score else "N/A"
                        source_file = node.metadata.get("file_name", "Unknown")

                        st.markdown(
                            f"**Source {i}** | Score: `{score:.4f}` | File: `{source_file}`"
                        )
                        st.markdown(
                            f"> {node.text[:500]}..."
                            if len(node.text) > 500
                            else f"> {node.text}"
                        )
                        st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": response.response}
        )


if __name__ == "__main__":
    main()
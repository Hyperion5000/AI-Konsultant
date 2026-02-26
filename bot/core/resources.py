import os
import logging
import pickle
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langgraph.graph.state import CompiledStateGraph

from bot import config
from bot.graph.workflow import create_agent_graph
from bot.graph.tools import (
    search_laws,
    calculate_penalty_214fz,
    calculate_penalty_zpp,
    set_retriever
)
from bot.core.logger import get_logger

logger = get_logger(__name__)

def initialize_bot_resources() -> Optional[CompiledStateGraph]:
    """
    Initializes all resources (LLM, Embeddings, Retrievers) and returns the compiled graph.
    """
    try:
        logger.info(f"Initializing resources with model {config.OLLAMA_MODEL}")

        # 1. Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

        if not os.path.exists(config.CHROMA_DIR):
            logger.error(f"Database directory {config.CHROMA_DIR} not found. Please run scripts/ingest_data.py first.")
            # We raise error to stop initialization if DB is missing
            raise FileNotFoundError(f"Database directory {config.CHROMA_DIR} not found.")

        vector_store = Chroma(
            persist_directory=config.CHROMA_DIR,
            embedding_function=embeddings
        )

        # 2. Retrievers
        chroma_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVER_K}
        )

        base_retriever = chroma_retriever

        # BM25
        bm25_path = os.path.join(config.BM25_DIR, "bm25_retriever.pkl")
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as f:
                    bm25_retriever = pickle.load(f)
                    bm25_retriever.k = config.RETRIEVER_K

                base_retriever = EnsembleRetriever(
                    retrievers=[chroma_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
                logger.info("EnsembleRetriever initialized.")
            except Exception as e:
                logger.error(f"Failed to load BM25 retriever: {e}. Falling back to Chroma only.")
        else:
            logger.warning(f"BM25 retriever not found at {bm25_path}. Using Chroma only.")

        # Reranker
        try:
            logger.info(f"Initializing Reranker: {config.RERANKER_MODEL}")
            model = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
            compressor = CrossEncoderReranker(model=model, top_n=config.RERANKER_K)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            final_retriever = compression_retriever
            logger.info("Reranker initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Reranker: {e}. Using base retriever.")
            final_retriever = base_retriever

        # 3. Configure Tools
        set_retriever(final_retriever)
        tools = [search_laws, calculate_penalty_214fz, calculate_penalty_zpp]

        # 4. LLM
        llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            temperature=config.TEMPERATURE,
            base_url=config.OLLAMA_BASE_URL
        )

        # 5. Create Graph
        graph = create_agent_graph(llm, tools)
        logger.info("Graph initialized successfully.")

        return graph

    except Exception as e:
        logger.error(f"Failed to initialize bot resources: {e}")
        raise

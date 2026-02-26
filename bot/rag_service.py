import os
import asyncio
import logging
import pickle
from typing import List, Dict, AsyncGenerator
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from bot import config
from bot.database import log_chat

# LangGraph imports
from bot.graph.workflow import create_agent_graph
from bot.graph.tools import (
    search_laws,
    calculate_penalty_214fz,
    calculate_penalty_zpp,
    set_retriever
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        try:
            logger.info(f"Initializing RAGService with model {config.OLLAMA_MODEL} and embeddings {config.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

            if not os.path.exists(config.CHROMA_DIR):
                logger.error(f"Database directory {config.CHROMA_DIR} not found. Please run create_index.py first.")
                raise FileNotFoundError(f"Database directory {config.CHROMA_DIR} not found.")

            self.vector_store = Chroma(
                persist_directory=config.CHROMA_DIR,
                embedding_function=self.embeddings
            )

            # Initialize Retrievers
            self.retriever = self._initialize_retriever()

            # Configure tools with the retriever
            set_retriever(self.retriever)

            self.llm = ChatOllama(
                model=config.OLLAMA_MODEL,
                temperature=config.TEMPERATURE,
                base_url=config.OLLAMA_BASE_URL
            )

            # Initialize the Agent Graph
            self.tools = [search_laws, calculate_penalty_214fz, calculate_penalty_zpp]
            self.graph = create_agent_graph(self.llm, self.tools)

            # Concurrency control
            self.llm_semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)

            # Simple in-memory history: user_id -> list of messages
            self.history: Dict[int, List[BaseMessage]] = {}
            logger.info("RAGService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}")
            raise

    def _initialize_retriever(self):
        """Initializes the hybrid retriever (Ensemble) + Reranker."""
        # Chroma Retriever
        chroma_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVER_K}
        )

        base_retriever = chroma_retriever

        # BM25 Retriever
        bm25_path = os.path.join(config.CHROMA_DIR, "bm25_retriever.pkl")
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as f:
                    bm25_retriever = pickle.load(f)
                    bm25_retriever.k = config.RETRIEVER_K

                logger.info("BM25 retriever loaded successfully.")

                # Create Ensemble
                # Weights: 0.5 for semantic, 0.5 for keyword
                base_retriever = EnsembleRetriever(
                    retrievers=[chroma_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
                logger.info("EnsembleRetriever initialized.")
            except Exception as e:
                logger.error(f"Failed to load BM25 retriever: {e}. Falling back to Chroma only.")
        else:
            logger.warning("BM25 retriever not found. Using Chroma only.")

        # Reranker
        try:
            logger.info(f"Initializing Reranker: {config.RERANKER_MODEL}")
            model = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
            compressor = CrossEncoderReranker(model=model, top_n=config.RERANKER_K)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            logger.info("Reranker initialized.")
            return compression_retriever
        except Exception as e:
            logger.error(f"Failed to initialize Reranker: {e}. Using base retriever.")
            return base_retriever

    def reset_history(self, user_id: int):
        """Clears the conversation history for a specific user."""
        if user_id in self.history:
            del self.history[user_id]
            logger.info(f"History cleared for user {user_id}")

    async def get_answer(self, user_id: int, question: str) -> AsyncGenerator[str, None]:
        """
        Generates an answer using the LangGraph Agent and streaming.
        Updates conversation history.
        """
        try:
            logger.info(f"Processing question for user {user_id}: {question}")

            # 1. Prepare history
            if user_id not in self.history:
                self.history[user_id] = []

            # Keep only last 3 pairs (6 messages)
            user_history = self.history[user_id][-6:]

            # 2. Construct messages
            # Updated system prompt for Agent
            system_prompt = (
                "Ты — опытный юрист-консультант и аналитик. "
                "Твоя цель — помочь пользователю, используя доступные инструменты.\n\n"
                "ПРАВИЛА:\n"
                "1. Сначала ПОДУМАЙ: какие инструменты нужны? (поиск законов, калькулятор).\n"
                "2. Если нужно найти информацию -> вызывай search_laws.\n"
                "3. Если нужно посчитать неустойку -> вызывай соответствующий калькулятор (214-ФЗ или ЗоЗПП).\n"
                "4. Отвечай СТРОГО на основе найденной информации и расчетов.\n\n"
                "СТРУКТУРА ОТВЕТА (обязательно используй Markdown):\n"
                "1. **Краткий вывод**: (Да / Нет / Сумма...). Четкий ответ.\n"
                "2. **Обоснование**: Подробное объяснение со ссылками на законы.\n"
                "3. **Источники**: Список использованных статей.\n\n"
                "Если инструмент вернул ошибку, исправь аргументы и попробуй снова."
            )

            messages = [SystemMessage(content=system_prompt)]
            messages.extend(user_history)
            messages.append(HumanMessage(content=question))

            input_state = {"messages": messages}

            full_response = ""

            # 3. Run Graph with Streaming
            # Use Semaphore to limit concurrent LLM usage
            async with self.llm_semaphore:
                # Use astream_events to capture tool usage and token streaming
                async for event in self.graph.astream_events(input_state, version="v2"):
                    kind = event["event"]

                    if kind == "on_tool_start":
                        # Log tool usage for user
                        yield "⏳ Выполняю запрос к инструментам...\n"

                    elif kind == "on_chat_model_stream":
                        # Stream tokens from the LLM
                        data = event["data"]
                        if "chunk" in data:
                            chunk = data["chunk"]
                            # Check if chunk has text content
                            if hasattr(chunk, "content") and chunk.content:
                                content = chunk.content
                                # Filter out empty strings
                                if isinstance(content, str) and content:
                                    full_response += content
                                    yield content

            # 4. Update history (Option B: Persistence via DB/Memory)
            self.history[user_id].append(HumanMessage(content=question))
            self.history[user_id].append(AIMessage(content=full_response))
            self.history[user_id] = self.history[user_id][-6:]

            # 5. Log analytics
            # Sources are now implicit in the answer, so we pass empty or generic info
            asyncio.create_task(log_chat(user_id, question, full_response, "Agent Managed"))

            logger.info(f"Finished generating response for user {user_id}")

        except Exception as e:
            logger.error(f"Error in get_answer: {e}")
            yield "Произошла ошибка при обработке вашего запроса."

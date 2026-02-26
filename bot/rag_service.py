import os
import logging
from typing import List, Dict, AsyncGenerator
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "cointegrated/rubert-tiny2")
DB_DIR = "db"
LLM_MODEL = "qwen2.5:7b"

class RAGService:
    def __init__(self):
        try:
            logger.info(f"Initializing RAGService with model {LLM_MODEL} and embeddings {EMBEDDING_MODEL_NAME}")
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

            if not os.path.exists(DB_DIR):
                logger.error(f"Database directory {DB_DIR} not found. Please run create_index.py first.")
                raise FileNotFoundError(f"Database directory {DB_DIR} not found.")

            self.vector_store = Chroma(
                persist_directory=DB_DIR,
                embedding_function=self.embeddings
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            self.llm = ChatOllama(
                model=LLM_MODEL,
                temperature=0.3,
                base_url="http://localhost:11434"
            )
            # Simple in-memory history: user_id -> list of messages
            self.history: Dict[int, List[BaseMessage]] = {}
            logger.info("RAGService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}")
            raise

    async def get_answer(self, user_id: int, question: str) -> AsyncGenerator[str, None]:
        """
        Generates an answer using RAG and streaming.
        Updates conversation history.
        """
        try:
            logger.info(f"Processing question for user {user_id}: {question}")

            # 1. Retrieve relevant documents
            docs = await self.retriever.ainvoke(question)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Retrieved {len(docs)} documents.")

            # 2. Prepare history
            if user_id not in self.history:
                self.history[user_id] = []

            # Keep only last 3 pairs (6 messages)
            user_history = self.history[user_id][-6:]

            # 3. Construct messages
            # Base system instruction
            messages = [
                SystemMessage(content="Ты — опытный юрист-консультант. Твоя задача — отвечать на вопросы пользователя "
                                      "СТРОГО на основе предоставленного текста законов. "
                                      "Если ответа нет, скажи об этом.")
            ]

            # Add history
            messages.extend(user_history)

            # Add current question with context
            # Adhering to the requested format: "Текст: {context}. Вопрос: {question}"
            final_user_content = f"Текст закона:\n{context_text}\n\nВопрос пользователя: {question}"
            messages.append(HumanMessage(content=final_user_content))

            # 4. Stream response
            full_response = ""
            async for chunk in self.llm.astream(messages):
                content = chunk.content
                if content:
                    full_response += content
                    yield content

            # 5. Update history
            # We store the *original* question in history, not the one with huge context,
            # to save tokens and keep history clean.
            self.history[user_id].append(HumanMessage(content=question))
            self.history[user_id].append(AIMessage(content=full_response))
            # Trim history again
            self.history[user_id] = self.history[user_id][-6:]

            logger.info(f"Finished generating response for user {user_id}")

        except Exception as e:
            logger.error(f"Error in get_answer: {e}")
            yield "Произошла ошибка при обработке вашего запроса."

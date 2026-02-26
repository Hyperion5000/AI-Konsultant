import asyncio
import logging
import time
from typing import AsyncGenerator

from aiogram import Router, F
from aiogram.types import Message
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from bot.core.prompts import (
    MSG_BOT_INITIALIZING,
    MSG_ANALYZING_LAWS,
    MSG_CONTINUE,
    MSG_NO_ANSWER,
    MSG_GENERATION_ERROR,
    MSG_GENERATION_ERROR_SHORT,
    CURSOR_MARKER,
    MSG_TOOL_USAGE,
    AGENT_SYSTEM_PROMPT
)
from bot.database import get_chat_history, log_chat
from bot.core.logger import get_logger

router = Router()
logger = get_logger(__name__)

@router.message(F.text)
async def handle_message(message: Message, graph: CompiledStateGraph, llm_lock: asyncio.Lock):
    """
    Handler for text messages.
    """
    if not graph:
        await message.answer(MSG_BOT_INITIALIZING)
        return

    user_id = message.from_user.id
    question = message.text

    await process_question(graph, message, user_id, question, llm_lock)

async def process_question(graph: CompiledStateGraph, message: Message, user_id: int, question: str, llm_lock: asyncio.Lock):
    """
    Process the question using LangGraph agent with streaming response.
    Handles long messages by splitting.
    """
    # Send initial status
    status_msg = await message.answer(MSG_ANALYZING_LAWS)

    full_response = ""       # Total response (cleaned)
    display_text = ""        # Text to display (including tool status)
    current_msg_obj = status_msg

    last_update_time = time.time()
    buffer = ""

    TELEGRAM_LIMIT = 4000

    try:
        # Acquire lock before starting generation to prevent CPU overload
        async with llm_lock:
            # 1. Fetch History
            history = await get_chat_history(user_id)

            # 2. Construct Messages
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
            messages.extend(history)
            messages.append(HumanMessage(content=question))

            input_state = {"messages": messages}

            # 3. Stream from Graph
            async for event in graph.astream_events(input_state, version="v2"):
                kind = event["event"]

                chunk_text = ""
                if kind == "on_tool_start":
                    chunk_text = MSG_TOOL_USAGE
                elif kind == "on_chat_model_stream":
                    data = event["data"]
                    if "chunk" in data:
                        chunk_obj = data["chunk"]
                        if hasattr(chunk_obj, "content") and chunk_obj.content:
                            content = chunk_obj.content
                            if isinstance(content, str) and content:
                                chunk_text = content
                                full_response += content

                if chunk_text:
                    display_text += chunk_text
                    buffer += chunk_text

                    # Check limits and update
                    if len(display_text) > TELEGRAM_LIMIT:
                        try:
                            await current_msg_obj.edit_text(display_text)
                        except Exception as e:
                            logger.warning(f"Failed to finalize message part: {e}")

                        current_msg_obj = await message.answer(MSG_CONTINUE)
                        display_text = ""
                        buffer = ""
                        last_update_time = time.time()
                        continue

                    current_time = time.time()
                    if (current_time - last_update_time > 1.5) and (len(buffer) > 30):
                        try:
                            await current_msg_obj.edit_text(display_text + CURSOR_MARKER)
                            last_update_time = current_time
                            buffer = ""
                        except Exception as e:
                            logger.warning(f"Failed to edit message: {e}")

        # Final update
        if display_text:
            try:
                await current_msg_obj.edit_text(display_text)
            except Exception as e:
                logger.warning(f"Failed to final edit message: {e}")
        elif not full_response:
             await current_msg_obj.edit_text(MSG_NO_ANSWER)

        # Log Chat
        asyncio.create_task(log_chat(user_id, question, full_response, "Agent Managed"))

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        try:
            await current_msg_obj.edit_text(MSG_GENERATION_ERROR)
        except:
            await message.answer(MSG_GENERATION_ERROR_SHORT)

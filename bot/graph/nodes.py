from typing import Any, Dict, List, Annotated
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode as CoreToolNode

from bot.graph.state import AgentState
from bot.core.prompts import (
    MSG_TOOL_ERROR_PREFIX,
    MSG_TOOL_ERROR_SUFFIX,
    MSG_TOOL_CRITICAL_ERROR
)

async def call_model(state: AgentState, config: RunnableConfig, llm: Runnable):
    """
    Executes the LLM.
    Args:
        state: The current graph state.
        config: The configuration for the run (callbacks, etc).
        llm: The Language Model runnable (already bound with tools).
    Returns:
        A dict with the new message from the LLM.
    """
    messages = state["messages"]
    # Pass the config to the LLM to propagate callbacks (e.g. tracing)
    response = await llm.ainvoke(messages, config)
    return {"messages": [response]}

def create_tool_node(tools: List[BaseTool]):
    """
    Creates a ToolNode that handles execution errors gracefully.
    If the LLM outputs invalid JSON for tool arguments (Pydantic ValidationError),
    this node catches the exception and returns a ToolMessage asking for correction.
    """
    # Initialize the standard ToolNode from langgraph.prebuilt
    tool_node = CoreToolNode(tools)

    async def safe_tool_node_func(state: AgentState, config: RunnableConfig):
        try:
            # Invoke the standard tool node with the config
            result = await tool_node.ainvoke(state, config)

            # Check if the result contains error messages
            # ToolNode returns dict with "messages" list
            if "messages" in result:
                new_messages = []
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage):
                        # Check for error status or content pattern
                        if msg.status == "error" or (msg.content and str(msg.content).startswith("Error:")):
                            # Replace with our custom error message
                            # Preserve the original error details for debugging/context if needed,
                            # but the prompt asks for a specific message to the LLM.
                            # "Ошибка вызова функции. Проверь аргументы и верни корректный JSON."
                            # We can append the original error to be helpful.
                            new_content = (
                                f"{MSG_TOOL_ERROR_PREFIX}{msg.content}"
                                f"{MSG_TOOL_ERROR_SUFFIX}"
                            )
                            new_msg = ToolMessage(
                                content=new_content,
                                tool_call_id=msg.tool_call_id,
                                status="error"
                            )
                            new_messages.append(new_msg)
                        else:
                            new_messages.append(msg)
                    else:
                        new_messages.append(msg)
                return {"messages": new_messages}

            return result

        except Exception as e:
            # Catch unexpected crashes (not handled by ToolNode)
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                tool_call = last_message.tool_calls[0]
                return {
                    "messages": [
                        ToolMessage(
                            content=f"{MSG_TOOL_CRITICAL_ERROR}{str(e)}"
                                    f"{MSG_TOOL_ERROR_SUFFIX}",
                            tool_call_id=tool_call["id"],
                            status="error"
                        )
                    ]
                }
            return {"messages": []}

    return safe_tool_node_func

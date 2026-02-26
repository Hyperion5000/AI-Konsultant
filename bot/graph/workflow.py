import functools
from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from bot.graph.state import AgentState
from bot.graph.nodes import call_model, create_tool_node

def create_agent_graph(llm: BaseChatModel, tools: List[BaseTool]):
    """
    Creates the LangGraph workflow for the agent.

    Args:
        llm: The initialized ChatOllama instance (not yet bound to tools).
        tools: List of tools the agent can use.

    Returns:
        A compiled StateGraph ready for execution.
    """
    # 1. Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # 2. Create the graph
    workflow = StateGraph(AgentState)

    # 3. Create nodes
    # 'agent': Calls the LLM
    agent_node = functools.partial(call_model, llm=llm_with_tools)

    # 'tools': Executes tool calls (with error handling wrapper)
    tool_node = create_tool_node(tools)

    # 4. Add nodes to graph
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 5. Define edges
    workflow.add_edge(START, "agent")

    # Conditional routing: If LLM calls tools -> go to 'tools', else -> END
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )

    # Loop back from tools to agent (ReAct loop)
    workflow.add_edge("tools", "agent")

    # 6. Compile
    return workflow.compile()

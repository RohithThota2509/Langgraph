import os
from typing import Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------
# 1. Define the System State
# ---------------------------------------------------------------------
# The state keeps track of the conversation history and the next step.
class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_agent: str

# ---------------------------------------------------------------------
# 2. Initialize the Local LLM via Ollama
# ---------------------------------------------------------------------
# Using llama3 as the default, but you can change this to mistral or phi3.
llm = Ollama(model="llama3", temperature=0)

# ---------------------------------------------------------------------
# 3. Define the Supervisor / Router
# ---------------------------------------------------------------------
# The supervisor reads the chat history and outputs the name of the next node.
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a supervisor managing a team of two specialized AI agents: MathAgent and CodingAgent.\n"
        "Your job is to look at the user request and decide who should handle it next.\n"
        "Rules:\n"
        "- If the request requires math, calculations, or logic puzzles, output exactly: MathAgent\n"
        "- If the request requires writing code, debugging, or software engineering advice, output exactly: CodingAgent\n"
        "- If the task is finished and the final answer is provided, output exactly: FINISH\n"
        "Respond with ONLY one of these three words: MathAgent, CodingAgent, or FINISH."
    )),
    MessagesPlaceholder(variable_name="messages"),
])

def supervisor_node(state: AgentState):
    # Format the prompt with the message history
    formatted_prompt = supervisor_prompt.format_messages(messages=state["messages"])
    response = llm.invoke(formatted_prompt)
    
    # Clean the response to ensure no extra whitespace/punctuation
    next_step = response.strip().replace(".", "").replace('"', '')
    
    # Fallback guardrail
    if next_step not in ["MathAgent", "CodingAgent", "FINISH"]:
        next_step = "FINISH"
        
    return {"next_agent": next_step}

# ---------------------------------------------------------------------
# 4. Define the Worker Agents
# ---------------------------------------------------------------------
def math_agent_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Math Agent. Solve the math problem clearly and step-by-step."),
        MessagesPlaceholder(variable_name="messages")
    ])
    response = llm.invoke(prompt.format_messages(messages=state["messages"]))
    # Append the agent's response to the conversation history
    return {"messages": [AIMessage(content=response, name="MathAgent")]}

def coding_agent_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Coding Agent. Provide efficient, clean code with brief explanations."),
        MessagesPlaceholder(variable_name="messages")
    ])
    response = llm.invoke(prompt.format_messages(messages=state["messages"]))
    return {"messages": [AIMessage(content=response, name="CodingAgent")]}

# ---------------------------------------------------------------------
# 5. Build the LangGraph Workflow
# ---------------------------------------------------------------------
workflow = StateGraph(AgentState)

# Add all nodes to the graph
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("MathAgent", math_agent_node)
workflow.add_node("CodingAgent", coding_agent_node)

# The Supervisor always runs first
workflow.add_edge(START, "Supervisor")

# After a worker finishes, control goes back to the Supervisor to review or finish
workflow.add_edge("MathAgent", "Supervisor")
workflow.add_edge("CodingAgent", "Supervisor")

# Define conditional routing logic based on the Supervisor's decision
def route_next(state: AgentState) -> Literal["MathAgent", "CodingAgent", "__end__"]:
    next_agent = state["next_agent"]
    if next_agent == "FINISH":
        return "__end__"
    return next_agent

workflow.add_conditional_edges(
    "Supervisor",
    route_next,
    {
        "MathAgent": "MathAgent",
        "CodingAgent": "CodingAgent",
        "__end__": END
    }
)

# Compile the graph into an executable application
app = workflow.compile()

# ---------------------------------------------------------------------
# 6. Execution Examples
# ---------------------------------------------------------------------
def run_system(user_query: str):
    print(f"\n{'='*60}\nUser Query: {user_query}\n{'='*60}")
    
    # Initialize state with the user message
    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }
    
    # Stream the graph execution to watch the routing happen live
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"\n[Node: {node_name}]")
            if "next_agent" in state_update:
                print(f"  Routing Decision -> Next Node: {state_update['next_agent']}")
            if "messages" in state_update:
                last_message = state_update["messages"][-1]
                print(f"  Response:\n{last_message.content}")

if __name__ == "__main__":
    # Test Case 1: Routes to Coding Agent
    run_system("Write a quick Python function to check if a word is a palindrome.")
    
    # Test Case 2: Routes to Math Agent
    run_system("What is the derivative of x^2 + 3x with respect to x?")
from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph  # Message Graph  is a state machine where messages flow between nodes. It handles message state and ensures the graph executes in the correct order.
from chains import ot_user_chain, ot_research_chain, ot_post_process_chain
import json

GENERATE = "generate_node"
RESEARCH = "research_node"
POST_PROCESS = "postprocess_node"


def generation_node(state: Sequence[BaseMessage]):
    return ot_user_chain.invoke({"messages": state})

def research_node(state: Sequence[BaseMessage]):
    return ot_research_chain.invoke({"messages": state})

def postprocess_node(state: Sequence[BaseMessage]):
    latest_message = state[-1].content  # Get the last message
    return ot_post_process_chain.invoke({"messages": [HumanMessage(content=latest_message)]})

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return POST_PROCESS
    elif state:  # Ensure there's always a defined exit path
        return RESEARCH
    return POST_PROCESS  # Fallback to prevent implicit exit

# Initialize the message graph (state machine for message processing)
builder = MessageGraph()

# Add nodes to the graph.
builder.add_node(GENERATE, generation_node)
builder.add_node(RESEARCH, research_node)
builder.add_node(POST_PROCESS, postprocess_node)

# Define the transitions between nodes.
builder.set_entry_point(GENERATE)  
builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(RESEARCH, GENERATE)
builder.add_edge(POST_PROCESS, END)

graph = builder.compile()

graph_structure = graph.get_graph(xray=True)
print(type(graph_structure))  # Debugging step

graph_image = graph_structure.draw_mermaid()
print(graph_image) # we can copy the output to https://mermaid.live/ to get the graph


if __name__ == '__main__':
    print ("Ready player one")
    inputs = HumanMessage(content=""""I have a 10-year-old patient with autism who is struggling with communication. The physician suspects a thumb issue, but the patient has sensory sensitivities to hand contact. 
                          I need suggestions for managing this, especially if a thumb splint is necessary."
                          """)
    response = graph.invoke(inputs)
    print(f"{response}")
    print(f"--------------------")

    last_message = response[-1].content
    print(f'{last_message}')
  
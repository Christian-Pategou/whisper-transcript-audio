from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from funct import GraphState
from funct import (
    retrieve_node,
    grade_documents_node,
    generate_node,
    transform_query_node,
    send_email_to_support_node,
    send_email_or_retry_cond,
    router_answer_cond
)


memory = MemorySaver()

# define the graph
def define_graph(memoire: InMemorySaver= memory) -> CompiledStateGraph:
    builder = StateGraph(GraphState)

    # define the node
    builder.add_node("retriever", retrieve_node)
    builder.add_node("grade_document", grade_documents_node)
    builder.add_node("generate", generate_node)
    builder.add_node("rewriter", transform_query_node)
    builder.add_node("send_email_to_support", send_email_to_support_node)

    # add edge
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "grade_document")
    builder.add_edge("grade_document", "generate")
    builder.add_conditional_edges(
        "generate",
        router_answer_cond,
        {
            "good": END,
            "bad": "rewriter",
            "humain": "send_email_to_support",
        }
    )

    builder.add_conditional_edges(
        "rewriter",
        send_email_or_retry_cond,
        {
            "email": "send_email_to_support",
            "not_email": "retriever",

        }
    )

    builder.add_edge("send_email_to_support", END)

    # compile the graph
    graph = builder.compile(checkpointer=memoire)

    return graph

def save_graph(graph:CompiledStateGraph, path:str)-> None:
    _ = graph.get_graph().draw_mermaid_png(output_file_path=path)

if __name__ == "__main__":
    graph = define_graph()
    save_graph(graph, "./graph.png")
    from pprint import pprint

    config = {
    "configurable": {
        "thread_id": "001"
    }
}
    # Run
    inputs = {
        "question": "je n'arrive plus a effectuer les consultations",
        "max_iter": 0
    }
    for output in graph.stream(inputs, config):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full GraphState at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["answer"])
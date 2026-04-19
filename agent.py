# agent.py

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from sentence_transformers import SentenceTransformer
import chromadb
import os

# -----------------------------
# SET GROQ API KEY
# -----------------------------
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -----------------------------
# STATE
# -----------------------------
class State(TypedDict):
    question: str
    messages: List[str]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int


# -----------------------------
# DB + EMBEDDINGS
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.Client()
collection = client.create_collection("papers")


def add_to_db(chunks, source_name="unknown"):
    embeddings = model.encode(chunks).tolist()
    
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": source_name}] * len(chunks)
    )


def retrieve_docs(query):
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    docs = results["documents"][0]
    sources = results["metadatas"][0]
    
    return docs, sources


# -----------------------------
# TOOL (Summarizer)
# -----------------------------
def summarize_tool(text):
    if not text:
        return "No content to summarize."
    
    response = llm.invoke([
        SystemMessage(content="Summarize the following text."),
        HumanMessage(content=text)
    ])
    
    return response.content


# -----------------------------
# NODES
# -----------------------------
def memory_node(state: State):
    msgs = state.get("messages", [])
    msgs.append(state["question"])
    return {"messages": msgs[-6:]}


# -----------------------------
# LLM ROUTER
# -----------------------------
def router_node(state: State):
    prompt = f"""
Decide the route for the question.

Routes:
- retrieve → for research paper questions
- tool → if user asks to summarize
- skip → greetings or casual chat

Reply ONLY with one word: retrieve / tool / skip

Question: {state['question']}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower()
    
    if route not in ["retrieve", "tool", "skip"]:
        route = "retrieve"
    
    return {"route": route}


# -----------------------------
# RETRIEVAL NODE
# -----------------------------
def retrieval_node(state: State):
    docs, sources = retrieve_docs(state["question"])
    context = "\n\n".join(docs)
    
    return {
        "retrieved": context,
        "sources": [s["source"] for s in sources]
    }


def skip_node(state: State):
    return {"retrieved": "", "sources": []}


# -----------------------------
# TOOL NODE
# -----------------------------
def tool_node(state: State):
    result = summarize_tool(state.get("retrieved", ""))
    return {"tool_result": result}


# -----------------------------
# ANSWER NODE (LLM)
# -----------------------------
def answer_node(state: State):
    context = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    
    system_prompt = """
You are a research assistant.

STRICT RULES:
- Answer ONLY from the provided context
- If answer not found → say "I don't know based on the provided documents"
- Do NOT hallucinate
"""

    user_prompt = f"""
Question: {state['question']}

Context:
{context}

Tool Output:
{tool_result}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return {"answer": response.content}


# -----------------------------
# EVAL NODE (LLM)
# -----------------------------
def eval_node(state: State):
    context = state.get("retrieved", "")
    
    if not context:
        return {"faithfulness": 1.0}
    
    prompt = f"""
Rate the faithfulness of the answer.

Score between 0 and 1.

Answer:
{state['answer']}

Context:
{context}

Only return a number.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        score = float(response.content.strip())
    except:
        score = 0.5
    
    retries = state.get("eval_retries", 0)
    
    return {
        "faithfulness": score,
        "eval_retries": retries + 1
    }


# -----------------------------
# SAVE NODE
# -----------------------------
def save_node(state: State):
    msgs = state.get("messages", [])
    msgs.append(state["answer"])
    return {"messages": msgs}


# -----------------------------
# ROUTING
# -----------------------------
def route_decision(state: State):
    return state["route"]


def eval_decision(state: State):
    if state["faithfulness"] < 0.7 and state["eval_retries"] < 2:
        return "answer"
    return "save"


# -----------------------------
# GRAPH
# -----------------------------
graph = StateGraph(State)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip", skip_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")

graph.add_edge("memory", "router")
graph.add_edge("retrieve", "answer")
graph.add_edge("skip", "answer")
graph.add_edge("tool", "answer")
graph.add_edge("answer", "eval")
graph.add_edge("save", END)

graph.add_conditional_edges("router", route_decision, {
    "retrieve": "retrieve",
    "tool": "tool",
    "skip": "skip"
})

graph.add_conditional_edges("eval", eval_decision, {
    "answer": "answer",
    "save": "save"
})

app = graph.compile(checkpointer=MemorySaver())


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def ask(question, thread_id="default"):
    result = app.invoke(
        {
            "question": question,
            "messages": [],
            "eval_retries": 0
        },
        config={"configurable": {"thread_id": thread_id}}
    )
    
    return result["answer"], result.get("sources", [])
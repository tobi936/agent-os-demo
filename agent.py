"""Demo LangGraph agent that reports lifecycle events to agent-os."""

import logging
import os
import sys

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from tracker import AgentOsTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    input: str
    classification: str
    result: str
    summary: str


# ---------------------------------------------------------------------------
# Tracker (module-level, initialised in main)
# ---------------------------------------------------------------------------

tracker: AgentOsTracker | None = None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify(state: AgentState) -> dict:
    """Classify the user input into a category."""
    tracker.step("classify", {"input": state["input"]})

    text = state["input"].lower()
    if any(w in text for w in ("bug", "error", "fix", "broken")):
        category = "bug_report"
    elif any(w in text for w in ("feature", "add", "new", "request")):
        category = "feature_request"
    else:
        category = "general_inquiry"

    logger.info("Classified as: %s", category)
    return {"classification": category}


def process(state: AgentState) -> dict:
    """Process the input based on its classification."""
    tracker.step("process", {"classification": state["classification"]})

    responses = {
        "bug_report": f"Bug report acknowledged: '{state['input']}'. Prioritised for review.",
        "feature_request": f"Feature request logged: '{state['input']}'. Added to backlog.",
        "general_inquiry": f"Inquiry received: '{state['input']}'. Routing to support.",
    }
    result = responses.get(state["classification"], "Unknown category.")
    logger.info("Processed: %s", result)
    return {"result": result}


def summarize(state: AgentState) -> dict:
    """Create a short summary of the run."""
    tracker.step("summarize", {"result": state["result"]})

    summary = (
        f"[{state['classification']}] {state['result']}"
    )
    logger.info("Summary: %s", summary)
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify)
    graph.add_node("process", process)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "process")
    graph.add_edge("process", "summarize")
    graph.add_edge("summarize", END)

    return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global tracker

    api_key = os.getenv("AGENT_OS_API_KEY")
    api_url = os.getenv("AGENT_OS_URL")

    if not api_key or not api_url:
        sys.exit("Set AGENT_OS_API_KEY and AGENT_OS_URL in .env")

    tracker = AgentOsTracker(api_url, api_key, agent_name="demo-agent")

    user_input = sys.argv[1] if len(sys.argv) > 1 else "There is a bug in the login flow"

    logger.info("Starting run %s", tracker.sdk_run_id)
    tracker.run_start({"input": user_input})

    try:
        app = build_graph().compile()
        result = app.invoke({"input": user_input})
        tracker.run_end({"summary": result.get("summary", "")})
        logger.info("Run completed successfully")
    except Exception as exc:
        tracker.error(str(exc))
        logger.exception("Run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

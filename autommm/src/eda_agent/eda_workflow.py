from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage
import yaml
from pathlib import Path
from datetime import datetime

# Import agents
from autommm.src.eda_agent.agents.eda_bot import data_overview, gen_data_profile, aggregator, formatter, sku_overview
from autommm.src.eda_agent.agents.infograph_bot import content_analyzer_agent, viz_planner_agent, frontend_coder_agent

# Define state type
class WorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    data_overview: Annotated[str, "Overview of the dataset"]
    data_profile: Annotated[str, "Detailed data profile"]
    sku_analysis: Annotated[dict, "Analysis for each SKU"]
    aggregated_insights: Annotated[str, "Combined insights from all analyses"]
    formatted_report: Annotated[str, "Final formatted report"]
    generated_html: Annotated[str, "The generated HTML content"]
    error_message: Annotated[str, "Any error message that occurred"]



def file_saver(state: WorkflowState) -> WorkflowState:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Markdown report
    if state.get("formatted_report"):
        md_path = output_dir / f"report_{timestamp}.md"
        md_path.write_text(state["formatted_report"], encoding="utf-8")

    # Save HTML output
    if state.get("generated_html"):
        html_path = output_dir / f"dashboard_{timestamp}.html"
        html_path.write_text(state["generated_html"], encoding="utf-8")

    return state  # Preserve state




workflow = StateGraph(WorkflowState)

# Add EDA nodes
workflow.add_node("data_overview_node", data_overview)
workflow.add_node("gen_data_profile", gen_data_profile)
workflow.add_node("sku_overview_a", lambda state: sku_overview(state, product="sku_a"))
workflow.add_node("sku_overview_b", lambda state: sku_overview(state, product="sku_b"))
workflow.add_node("sku_overview_c", lambda state: sku_overview(state, product="sku_c"))
workflow.add_node("aggregator", aggregator)
workflow.add_node("formatter", formatter)

# Add visualization nodes
workflow.add_node("analyze_content", content_analyzer_agent)
workflow.add_node("plan_viz", viz_planner_agent)
workflow.add_node("generate_html", frontend_coder_agent)

# Define workflow edges
# Start with parallel EDA tasks
workflow.add_edge(START, "data_overview_node")
workflow.add_edge(START, "gen_data_profile")
workflow.add_edge(START, "sku_overview_a")
workflow.add_edge(START, "sku_overview_b")
workflow.add_edge(START, "sku_overview_c")

# Aggregate EDA results
workflow.add_edge("data_overview_node", "aggregator")
workflow.add_edge("sku_overview_a", "aggregator")
workflow.add_edge("sku_overview_b", "aggregator")
workflow.add_edge("sku_overview_c", "aggregator")
workflow.add_edge("aggregator", "formatter")

# Flow into visualization
workflow.add_edge("formatter", "analyze_content")
workflow.add_edge("analyze_content", "plan_viz")
workflow.add_node("file_saver", file_saver)

workflow.add_edge("plan_viz", "generate_html")
workflow.add_edge("generate_html", "file_saver")
workflow.add_edge("formatter", "file_saver")  

workflow.add_edge("file_saver", END)
workflow.add_edge("gen_data_profile", END)

# Compile the workflow
workflow = workflow.compile()


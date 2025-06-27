from langgraph.graph import StateGraph, END
from agents.src import data_analysis_agent, InsightGenerator, ModelConfigurator, ModelExecutor, ContributionEvaluator, BusinessInterpreter, QualityModerator
from memory.memory_store import memory
from langchain.schema import BaseOutputParser


class AgentState(dict):
    pass

# Routing logic
def route_node(state: AgentState):
    feedback = state.get("user_feedback", "")
    if "back to analysis" in feedback.lower():
        return "data_analysis"
    elif "approve" in feedback.lower():
        return state.get("next_node", "END")
    return "quality_debate"

# Define graph
builder = StateGraph(AgentState)

builder.add_node("data_analysis", data_analysis.run)
builder.add_node("data_insights", data_insights.run)
builder.add_node("model_preparation", model_preparation.run)
builder.add_node("model_runner", model_runner.run)
builder.add_node("contribution_calc", contribution_calc.run)
builder.add_node("business_insights", business_insights.run)
builder.add_node("quality_debate", quality_debate.run)

# Routing
for node in ["data_analysis", "data_insights", "model_preparation", "model_runner", "contribution_calc", "business_insights"]:
    builder.add_edge(node, "quality_debate")
    builder.add_conditional_edges("quality_debate", route_node)

builder.set_entry_point("data_analysis")
builder.add_edge("business_insights", END)

graph = builder.compile()

# Run the graph
result = graph.invoke({"memory": memory, "user_feedback": ""})
print(result)

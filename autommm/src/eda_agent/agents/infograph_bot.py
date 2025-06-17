import os
from autommm.config.process_configuration import process_config
from autommm.config import configuration

from langchain_core.tools import tool
import pandas as pd
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Optional, Union, Any
import operator
from pathlib import Path

import json
import yaml
from IPython.display import Markdown, Image
from langchain_experimental.utilities import PythonREPL

from langgraph.graph import START, END, StateGraph
import subprocess


config = process_config(configuration)

df = config["master_data"]
llm_infograph = config["llm_infograph"]
llm = config["llm"]
data_description = config["data_description"]
python310_executable = config["python310_executable"]
data_profile_path = config["data_profile_path"]


prompts_config_path = os.path.join("autommm", "src", "eda_agent", "prompts_config.yaml")
def load_prompt_config(path: str, key: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)[key]


class GraphState(TypedDict):
    raw_text: str  # Initial input: the text context for the infographic
    extracted_content: Optional[Dict]  # Output from ContentAnalyzerAgent (JSON)
    visualization_plan: Optional[Dict]  # Output from VisualizationPlannerAgent (JSON)
    generated_html: Optional[str]  # Output from FrontendCoderAgent (HTML string)
    error_message: Optional[str]  # Optional: for handling errors in any step


class DataPoint(BaseModel):
    type: str
    description: str
    value: Union[str, int, float, List[Union[str, int, float]], dict]
    context_text: Optional[str] = None


class Section(BaseModel):
    heading: str
    intro_text: str
    data_points: List[DataPoint]


class InfographicData(BaseModel):
    infographic_title: str
    sections: List[Section]


def content_analyzer_agent(state: GraphState) -> Dict:
    content_analyzer_llm = llm.with_structured_output(InfographicData)
    print("--- ContentAnalyzerAgent: Analyzing text with LLM... ---")
    raw_text = state.get("raw_text", "")
    print("--- Node: Generating overall data overview... ---")
    prompt_config = load_prompt_config(
        prompts_config_path, "content_structuring_agent"
    )
    user_prompt = prompt_config["template"].format(raw_text=raw_text)
    messages = [
        {"role": "system", "content": f"goal: {prompt_config['goal']}"},
        {"role": "system", "content": f"backstory: {prompt_config['backstory']}"},
        {"role": "system", "content": f"instruction: {prompt_config['instruction']}"},
        {"role": "user", "content": user_prompt},
    ]
    try:
        llm_output = content_analyzer_llm.invoke(messages)
        extracted_data = json.loads(llm_output.model_dump_json())
        print("Content analysis complete.")
        # print("extracted : ", extracted_data)
        return {"extracted_content": extracted_data, "error_message": None}
    except Exception as e:
        print(f"Error in ContentAnalyzerAgent: {e}")
        return {"error_message": f"ContentAnalyzer failed: {e}"}


class VizPlan(BaseModel):
    type: str  # e.g., 'Chart.js_Line', 'HTML_Flowchart', etc.
    chart_data: Optional[dict]  = {}# Optional, can be None for non-chart visualizations
    layout_tailwind_classes: str  # Tailwind CSS grid layout classes
    explanation_text: str  # Max 80 words explanation of the visualization


class VizDataPoint(BaseModel):
    type: str  # e.g., 'statistic', 'trend', 'category'
    description: str  # A short descriptor of the data
    value: Any  # Can be int, float, str, etc.
    context_text: Optional[str] = None  # Optional additional context
    viz_plan: VizPlan  # Visualization details


class VizSection(BaseModel):
    heading: str  # Section title
    intro_text: str  # Brief introduction to the section
    data_points: List[VizDataPoint]  # List of data points in the section


class VizPlannerOutput(BaseModel):
    infographic_title: str  # Overall infographic title
    sections: List[VizSection]  # List of sections
    overall_layout_strategy: str  # High-level layout description


def viz_planner_agent(state: GraphState) -> Dict:
    """
    VisualizationPlannerAgent: Takes structured content and plans the visualizations using an LLM.
    """
    viz_planner_llm = llm.with_structured_output(VizPlannerOutput)

    print("--- VisualizationPlannerAgent: Planning visuals with LLM... ---")
    extracted_content = state.get("extracted_content")

    if not extracted_content:
        return {"error_message": "No extracted content to plan visualizations for."}

    viz_planner_prompt = f"""
    You are an expert Infographic UI/UX Designer and Data Storyteller. Your goal is to design the visual representation for an infographic based on the provided structured content analysis. You must select the most appropriate visualization types and plan the layout, strictly adhering to the specified technical and design constraints.

    **Input Content Analysis (JSON):**
    ```json
    {json.dumps(extracted_content, indent=2)}
    ```

    **Infographic Chart Selection Guide & Constraints (Your Knowledge Base):**
    * **Goal: Inform (Convey a single important data point)**
        * Single Big Number: Use large, bold text.
        * Donut/Pie Chart: Simple proportion (Chart.js).
    * **Goal: Compare (Compare categories or show composition)**
        * Bar Chart (Chart.js): Compare values across many categories.
        * Bubble Chart (Chart.js): Compare values across a few categories (for 3 variables).
        * Stacked Bar Chart (Chart.js): Show composition within categories.
    * **Goal: Change (Show change over time)**
        * Line Chart (Chart.js): Show trends.
        * Area Chart (Chart.js): Show trends, emphasize volume.
        * Timeline: Show distinct events (Structured HTML/CSS with Tailwind).
    * **Goal: Organize (Show groupings, rankings, processes)**
        * List/Table: Standard HTML (`<ul>`, `<table>`).
        * Flow Chart: Show complex processes (Structured HTML/CSS with Tailwind).
        * Radar Chart (Chart.js): Compare multiple metrics for an entity.
    * **Goal: Relationships (Reveal correlations or distributions)**
        * Scatter Plot (Chart.js): Show relationship between two variables.

    * **CRITICAL: NO SVG GRAPHICS ARE ALLOWED ANYWHERE.**
    * **CRITICAL: NO MERMAID JS IS ALLOWED FOR DIAGRAMS.**
    * **Chart Library:** Primarily use Chart.js for all numerical charts.
    * **Diagrams:** MUST be built exclusively with structured HTML/CSS and Tailwind utilities (e.g., for boxes, lines, arrows).
    * **Color Palette (Energetic & Playful - use these HEX codes for charts):**
        * Main Chart Colors (cyclical): `#FF6B6B`, `#FFD166`, `#06D6A0`, `#118AB2`, `#073B4C`.
        * Backgrounds/Text: Use Tailwind's `slate-50`, `gray-800`, `#073B4C`, `#118AB2`.

    **Output Format (JSON - enhanced version of input):**
    ```json
    {{
      "infographic_title": "string",
      "sections": [
        {{
          "heading": "string",
          "intro_text": "string",
          "data_points": [
            {{
              "type": "string",
              "description": "string",
              "value": "any",
              "context_text": "string (optional)",
              "viz_plan": {{
                "type": "string (e.g., 'Chart.js_Line', 'HTML_Flowchart', 'BigNumber', 'Chart.js_Radar')",
                "chart_data": "object (e.g., {{labels:[], datasets:[{{label:'', data:[], backgroundColor:''}}]}} or null for non-chart viz)",
                "layout_tailwind_classes": "string (e.g., 'col-span-1 md:col-span-1 lg:col-span-1', 'md:col-span-2' for wider elements)",
                "explanation_text": "string (concise explanation of what the visualization shows and its key takeaway from the context, max 80 words)"
              }}
            }}
          ]
        }}
      ],
      "overall_layout_strategy": "string (brief high-level layout description, e.g., 'responsive grid, 1-2 columns, cards for sections')"
    }}
    ```
    **Instructions:**
    1.  Review each 'section' and 'data_point' from the input JSON.
    2.  For each 'data_point', generate a 'viz_plan':
        * Select the most appropriate 'type' from the "Infographic Chart Selection Guide" and list it as `Chart.js_[Type]` or `HTML_[Type]` or `BigNumber`.
        * If it's a Chart.js visualization, populate 'chart_data' with realistic, well-formatted data (labels, datasets, colors from the palette).
        * Determine 'layout_tailwind_classes' to define its size and position within the grid for responsiveness. Prioritize using `md:col-span-2` for diagrams or more complex charts.
        * Write a concise 'explanation_text' to accompany the visualization on the infographic.
    3.  Ensure the entire output is a single, valid, complete JSON object. Do NOT include any explanations or conversational text outside the JSON.
    """
    try:
        llm_output = viz_planner_llm.invoke([{"role": "user", "content": viz_planner_prompt}])
        viz_plan = json.loads(llm_output.model_dump_json())
        print("Visualization planning complete.")
        print("viz plan: ", viz_plan)
        return {"visualization_plan": viz_plan, "error_message": None}
    except Exception as e:
        print(f"Error in VisualizationPlannerAgent: {e}")
        return {"error_message": f"VisualizationPlanner failed: {e}"}

def frontend_coder_agent(state: GraphState) -> Dict:
    """
    FrontendCoderAgent: Generates the full HTML infographic based on the viz plan using an LLM.
    """
    print("--- FrontendCoderAgent: Generating HTML with LLM... ---")
    viz_plan = state.get("visualization_plan")

    if not viz_plan:
        return {"error_message": "No visualization plan to generate HTML from."}

    # Prompt for FrontendCoderAgent
    prompt_config = load_prompt_config(
        prompts_config_path, "frontend_coder_agent"
    )
    
    frontend_coder_prompt = f"""
    **goal:** {prompt_config['goal']}  
    **backstory:** {prompt_config['backstory']}
    **Instructions:** {prompt_config['instruction']}
    **template:**
    **Input Visualization Plan (JSON):**
        ```json
        {json.dumps(viz_plan, indent=2)}
        ```
    """

    try:
        generated_html_code = llm_infograph.invoke([{"role": "user", "content": frontend_coder_prompt}])
        print("HTML generation complete.")
        # print("html code: ", generated_html_code.content)
        return {"generated_html": generated_html_code.content, "error_message": None}
    except Exception as e:
        print(f"Error in FrontendCoderAgent: {e}")
        return {"error_message": f"FrontendCoder failed: {e}"}


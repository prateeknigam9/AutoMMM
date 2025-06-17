import os
from autommm.config.process_configuration import process_config
from autommm.config import configuration

from langchain_core.tools import tool
import pandas as pd
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated, List,  Dict, Optional, Union, Any
import operator
from pathlib import Path

import json

from IPython.display import Markdown, Image
from langchain_experimental.utilities import PythonREPL

from langgraph.graph import START, END, StateGraph
import subprocess


config = process_config(configuration)

df = config['master_data'] 
llm_infograph = config['llm_infograph']
llm = config['llm']
data_description = config['data_description']
python310_executable = config['python310_executable']
data_profile_path = config['data_profile_path']


# --- 1. Define the Graph State ---
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

    content_analyzer_llm = llm_infograph.with_structured_output(InfographicData)

    print("--- ContentAnalyzerAgent: Analyzing text with LLM... ---")
    raw_text = state.get("raw_text", "")

    # Prompt for ContentAnalyzerAgent
    content_analyzer_prompt = f"""
    You are an expert Data Analyst and Content Summarizer. Your task is to meticulously read the provided text and extract all relevant information, structuring it into a precise JSON format for subsequent visualization. Do not interpret or design any visuals; focus solely on accurate content analysis and structuring.

    **Input Text:**
    {raw_text}

    **Output Format (JSON Schema):**
    ```json
    {{
      "infographic_title": "string (concise, catchy title based on main topic, max 8 words)",
      "sections": [
        {{
          "heading": "string (main heading for this section, max 10 words)",
          "intro_text": "string (brief introductory paragraph for this section, max 100 words)",
          "data_points": [
            {{
              "type": "string (e.g., 'KPI', 'Trend', 'Comparison', 'ProcessStep', 'Definition', 'TimelineEvent', 'StatisticalSummary', 'KeyInsight')",
              "description": "string (brief explanation of what this data point represents, max 20 words)",
              "value": "any (numeric value, array of strings/numbers, object, or string depending on type. For processes, use an array of strings for steps.)",
              "context_text": "string (short snippet of original text for context, max 50 words, optional)"
            }}
          ]
        }}
      ]
    }}
    ```
    **Instructions:**
    1. Read the entire text carefully to grasp the overall context.
    2. Identify distinct logical sections within the text. For each section, create a 'heading' and a brief 'intro_text'.
    3. Within each section, identify all potential data points suitable for visualization. Extract them accurately.
    4. For each 'data_point':
        * Assign the most appropriate 'type' from the suggested list.
        * Write a concise 'description'.
        * Extract its 'value'. Ensure numeric values are actual numbers, arrays are actual arrays, etc.
        * Provide 'context_text' if a short direct quote or reference from the original text helps.
    5. Ensure the entire output is a single, valid, complete JSON object. Do NOT include any explanations or conversational text outside the JSON.
    """

    try:
        llm_output = content_analyzer_llm.invoke([{"role": "user", "content": content_analyzer_prompt}])
        extracted_data = json.loads(llm_output.model_dump_json())
        print("Content analysis complete.")
        print("extracted : ", extracted_data)
        return {"extracted_content": extracted_data, "error_message": None}
    except Exception as e:
        print(f"Error in ContentAnalyzerAgent: {e}")
        return {"error_message": f"ContentAnalyzer failed: {e}"}



class VizPlan(BaseModel):
    type: str  # e.g., 'Chart.js_Line', 'HTML_Flowchart', etc.
    chart_data: Optional[dict]  # Optional, can be None for non-chart visualizations
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
    frontend_coder_prompt = f"""
    You are a meticulous Frontend Developer and Code Generator specializing in single-page HTML infographics. Your task is to generate a complete, single HTML file infographic based on the provided visualization plan. You must strictly adhere to all technical, styling, and library-specific constraints. Ensure the entire HTML is valid and self-contained.

    **Input Visualization Plan (JSON):**
    ```json
    {json.dumps(viz_plan, indent=2)}
    ```

    **Technical & Styling Constraints (Your Hard Rules):**
    1.  **Output:** A single, complete, valid HTML5 file string. Do NOT include any other text or explanations outside the HTML.
    2.  **CSS Framework:** Tailwind CSS (CDN: `<script src="https://cdn.tailwindcss.com"></script>`). Use Tailwind utility classes for all styling.
    3.  **Font:** Use 'Inter' font from Google Fonts (CDN link in head).
    4.  **Chart Library:** Chart.js (CDN: `<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>`). ALL charts MUST be Chart.js and render to Canvas.
    5.  **Diagrams/Timelines:** MUST be built exclusively with structured HTML/CSS using Tailwind. **NO SVG, NO Mermaid JS.** Use Unicode characters for arrows (e.g., `&darr;`, `&rArr;`) or simple CSS shapes.
    6.  **Chart Container Styling:** Every Chart.js `<canvas>` MUST be wrapped in a `<div>` with the class `chart-container`. Define this class in an embedded `<style>` block in the HTML `<head>` with these exact properties for responsiveness and size control:
        ```css
        .chart-container {{
            position: relative;
            width: 100%;
            max-width: 600px; /* Base max width, adjust as needed in Tailwind */
            margin-left: auto;
            margin-right: auto;
            height: 320px; /* Base height */
            max-height: 400px;
        }}
        @media (min-width: 768px) {{ .chart-container {{ height: 350px; }} }}
        .flow-node {{ border: 2px solid #118AB2; background-color: #ffffff; color: #073B4C; border-radius: 0.5rem; padding: 0.75rem; }}
        .arrow-down {{ width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 8px solid #118AB2; margin: 0.5rem auto; }}
        ```
    7.  **Chart.js Specifics:**
        * Set `options.responsive: true` and `options.maintainAspectRatio: false`.
        * **Label Wrapping:** Any string label in `labels` array longer than 16 characters MUST be processed into an array of strings. Split words to create lines, keeping each line around 16 chars. Example: `'Long Label Example String'` -> `['Long Label', 'Example String']`.
        * **Tooltip Callback:** ALL Chart.js instances MUST include this exact configuration within their `options.plugins.tooltip.callbacks` object to handle wrapped labels:
            ```javascript
            title: function(tooltipItems) {{
                const item = tooltipItems[0];
                let label = item.chart.data.labels[item.dataIndex];
                return Array.isArray(label) ? label.join(' ') : label;
            }}
            ```
    8.  **Color Palette (Energetic & Playful):** Apply these consistently.
        * Background: `bg-slate-50` (`#f8fafc`)
        * Main Text/Headings: `text-[#073B4C]` (Dark Blue)
        * Accent/Gradient Start: `text-[#118AB2]` (Blue-Green)
        * Accent/Gradient End: `text-[#06D6A0]` (Bright Green)
        * Card Background: `bg-white` (`#ffffff`)
        * Chart Colors (for `backgroundColor` and `borderColor` of datasets, cycle through these):
            * `#FF6B6B` (Red)
            * `#FFD166` (Yellow)
            * `#06D6A0` (Bright Green)
            * `#118AB2` (Blue-Green)
            * `#073B4C` (Dark Blue)
            * For lighter fills, use `rgba(R,G,B, 0.1)` or `0.2`.
    9.  **Design Principles:** Use Material Design aesthetics (cards with shadows, clear typography hierarchy, intuitive spacing).
    10. **Content Integration:** Populate all section headings, introductory paragraphs, and detailed explanation text (`viz_plan.explanation_text`) for each visualization directly from the provided plan.
    11. **Structure:** Use a `container mx-auto`, `grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3` for the main content layout. Each section should be a `card`.
    12. **No Comments:** Do NOT include any HTML, CSS, or JavaScript comments in the final generated HTML string.

    Generate the complete, runnable HTML for the infographic now.
    """

    try:
        generated_html_code = llm_infograph.invoke([{"role": "user", "content": frontend_coder_prompt}])
        print("HTML generation complete.")
        print("html code: ", generated_html_code.content)
        return {"generated_html": generated_html_code.content, "error_message": None}
    except Exception as e:
        print(f"Error in FrontendCoderAgent: {e}")
        return {"error_message": f"FrontendCoder failed: {e}"}
    
    

# Create the graph instance
workflow = StateGraph(GraphState)

# Add nodes (agents) to the graph
workflow.add_node("analyze_content", content_analyzer_agent)
workflow.add_node("plan_viz", viz_planner_agent)
workflow.add_node("generate_html", frontend_coder_agent)

# This is a linear flow: Analyze -> Plan -> Generate
workflow.add_edge(START, "analyze_content")
workflow.add_edge("analyze_content", "plan_viz")
workflow.add_edge("plan_viz", "generate_html")
workflow.add_edge("generate_html", END)


infograph_workflow = workflow.compile()
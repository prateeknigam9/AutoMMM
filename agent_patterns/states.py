import operator
from typing import Annotated, TypedDict, List, Literal, Any
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing import Union

class ToolAgentState(TypedDict):
    query: str
    message: Annotated[List[AnyMessage], operator.add]
    task_tool_pairs: Any
    tool_response_list: List
    user_suggestions: List[str]
    finalresponse:str

class Feedback(BaseModel):
    category: Literal['approve', 'retry', 'retry with suggestion']
    feedback : str = Field(..., description="user response")
    thought: str = Field(..., description="Reasoning about what needs to be done next")

class DataValidationState(TypedDict):
    column_categories: dict
    distinct_products : list
    user_feedback : Feedback | None
    node_to_update: str
    data_val_report : str
    tool_results: List[Union[str, dict]]
    messages : Annotated[List[AnyMessage], operator.add]
    data_summary: dict
    final_report:str



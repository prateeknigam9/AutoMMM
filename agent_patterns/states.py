import operator
from typing import Annotated, TypedDict, List, Literal, Any, Optional
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing import Union

class ToolAgentState(TypedDict):
    query: str
    RephrasedQuery : str
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
    completed : bool

class DataEngineerState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    completed : bool

class DataAnalystState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    data_summary: dict
    column_categories: dict
    user_feedback : Feedback | None
    distinct_products : list[str]
    completed : bool

class DataQualityAnalystState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    tool_results: List[Union[str, dict]]
    qa_report : str
    qa_report_path : str
    completed : bool

class DataInsightState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]

class DataTeamManagerState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    data_loaded : bool
    analysis_done : bool
    quality_assurance :bool
    data_analysis_report : dict
    qa_report : dict
    task : str
    next_agent : str
    command : Literal['chat','run','start',None]


 

class modelConfigSchema(BaseModel):
    kpi : List[str]
    prior_mean : List[int]
    prior_sd : List[int]
    is_random : List[Literal[0,1]]
    lower_bound : List[float]
    upper_bound: List[float]
    compute_contribution : List[Literal[0,1]]	

class ModellingTeamManagerState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    task : str
    meta_model_config: dict
    model_config : modelConfigSchema
    config_interpreter : str
    performance_analyst : str
    coef_explainer: str
    tuning_recommender: str
    final_report: str

class ConfigurationArchitectState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    meta_model_config: dict
    model_config : modelConfigSchema

class ModelEvaluatorState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    meta_model_config: dict
    model_config : modelConfigSchema
    config_interpreter : str
    performance_analyst : str
    coef_explainer: str
    tuning_recommender: str
    final_report: str

# Contribution Team States
class ContributionAnalystState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    analysis_type: str
    contribution_analysis_results: dict
    contribution_report: dict
    completed : bool

class ContributionInterpreterState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    interpretation_type: str
    contribution_interpretation_results: dict
    business_insights: list
    actionable_recommendations: list
    marketing_optimization: dict
    completed : bool

class ContributionValidatorState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    validation_type: str
    validation_results: dict
    validation_report: dict
    validation_report_path: str
    completed : bool

class ContributionTeamManagerState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    contribution_analysis_done : bool
    contribution_interpretation_done : bool
    contribution_validation_done : bool
    contribution_analysis_report : dict
    contribution_interpretation_report : dict
    contribution_validation_report : dict
    task : str
    next_agent : str
    command : Literal['chat','run','start',None]

# CEO State - Manages Complete Flow
class CEOState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    data_team_status : bool
    modelling_team_status : bool
    contribution_team_status : bool
    overall_project_status : str
    current_phase : str
    next_phase : str
    data_team_report : dict
    modelling_team_report : dict
    contribution_team_report : dict
    final_executive_summary : str
    task : str
    next_team : str
    command : Literal['chat','run','start','overview','status',None]
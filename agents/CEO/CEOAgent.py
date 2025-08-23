from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from agent_patterns.states import CEOState
from utils import chat_utility
from utils import theme_utility
from utils.memory_handler import DataStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.prompts.chat import ChatPromptTemplate
from rich import print
from utils import utility

# Import all team managers
from agents.DataHandlingTeam.DataTeamManager import DataTeamManagerAgent
from agents.ModelRunnerTeam.ModellingTeamManager import ModellingTeamManagerAgent
from agents.ContributionTeam.ContributionTeamManager import ContributionTeamManagerAgent

# Load prompts
CEOAgentPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "ceo_agent_prompts",
)

user_messages = utility.load_prompt_config(
    r"prompts\user_messages.yaml",
    "ceo_agent",
)

checkpointer = InMemorySaver()


class CEOAgent:
    def __init__(self, agent_name: str, agent_description: str, backstory: str = ""):
        self.agent_name = f"{agent_name}: CEO"
        self.agent_description = agent_description
        self.backstory = backstory
        self.llm = ChatOllama(model="llama3.1")
        self.openaillm = ChatOpenAI(model="gpt-4o-mini")
        
        # Initialize all team managers
        self.data_team_manager = DataTeamManagerAgent(
            agent_name="Data Team Manager",
            agent_description="Manages data loading, analysis, and quality assurance",
            backstory="Expert in data preparation and validation for marketing mix modeling"
        )
        
        self.modelling_team_manager = ModellingTeamManagerAgent(
            agent_name="Modelling Team Manager",
            agent_description="Manages model configuration, execution, and evaluation",
            backstory="Specialist in hierarchical Bayesian regression modeling and optimization"
        )
        
        self.contribution_team_manager = ContributionTeamManagerAgent(
            agent_name="Contribution Team Manager",
            agent_description="Manages marketing contribution analysis and business insights",
            backstory="Expert in marketing attribution, ROI analysis, and strategic recommendations"
        )
        
        # Build the CEO workflow graph
        self.graph = self.build_graph(CEOState)
    
    def chatNode(self, state: CEOState):
        prompt = CEOAgentPrompt['ceo_chat_prompt'].format(
            agent_name=self.agent_name,
            agent_description=self.agent_description,
            backstory=self.backstory,
            data_team_status=state.get('data_team_status', False),
            modelling_team_status=state.get('modelling_team_status', False),
            contribution_team_status=state.get('contribution_team_status', False),
            overall_project_status=state.get('overall_project_status', 'Not Started'),
            current_phase=state.get('current_phase', 'Initialization')
        )
        
        state['messages'] = state['messages'] + [
            chat_utility.build_message_structure(role="system", message=prompt)
        ]
        
        take_input_prompt = f"Hello! I'm {self.agent_name}, CEO of AutoMMM. How can I help you today?"
        
        while True:
            user_input = chat_utility.take_user_input(take_input_prompt)
            state['messages'] = state['messages'] + [
                chat_utility.build_message_structure(role="user", message=user_input)
            ]
            take_input_prompt = "USER"
            
            if user_input.lower() == "exit":
                break
            
            response = self.llm.invoke(state['messages'])
            theme_utility.display_response(response.content, title=self.agent_name)
            
            parsed = chat_utility.parse_json_from_response(response.content)
            if parsed and isinstance(parsed, dict):
                if "call_team" in parsed:
                    team = parsed['call_team']
                    task = parsed.get('task', 'Execute team workflow')
                    state['task'] = task
                    state['next_team'] = team
                    
                    if team == "data_team":
                        return Command(
                            goto="data_team_node",
                            update={
                                "messages": [chat_utility.build_message_structure(
                                    role="assistant", 
                                    message=f"Executing Data Team workflow: {task}"
                                )]
                            }
                        )
                    elif team == "modelling_team":
                        return Command(
                            goto="modelling_team_node",
                            update={
                                "messages": [chat_utility.build_message_structure(
                                    role="assistant", 
                                    message=f"Executing Modelling Team workflow: {task}"
                                )]
                            }
                        )
                    elif team == "contribution_team":
                        return Command(
                            goto="contribution_team_node",
                            update={
                                "messages": [chat_utility.build_message_structure(
                                    role="assistant", 
                                    message=f"Executing Contribution Team workflow: {task}"
                                )]
                            }
                        )
                    elif team == "overview":
                        return Command(
                            goto="project_overview_node",
                            update={
                                "messages": [chat_utility.build_message_structure(
                                    role="assistant", 
                                    message="Generating project overview"
                                )]
                            }
                        )
                    elif team == "executive_summary":
                        return Command(
                            goto="executive_summary_node",
                            update={
                                "messages": [chat_utility.build_message_structure(
                                    role="assistant", 
                                    message="Creating executive summary"
                                )]
                            }
                        )
                    elif team == "__end__":
                        return Command(
                            goto="__end__",
                            update={
                                "messages": [chat_utility.build_message_structure(
                                    role="assistant", 
                                    message="Workflow completed. Thank you!"
                                )]
                            }
                        )
            
            # If no specific command, continue chat
            state['messages'] = state['messages'] + [
                chat_utility.build_message_structure(role="assistant", message=response.content)
            ]
        
        return Command(
            goto="__end__",
            update={
                "messages": [chat_utility.build_message_structure(
                    role="assistant", 
                    message="CEO session ended. Thank you!"
                )]
            }
        )
    
    def data_team_node(self, state: CEOState):
        """Execute Data Team workflow"""
        theme_utility.display_response("Executing Data Team workflow...", title=self.agent_name)
        
        # Initialize data team state
        data_team_state = {
            'messages': [{"role": "user", "content": state.get('task', 'Execute data workflow')}],
            'data_loaded': False,
            'analysis_done': False,
            'quality_assurance': False,
            'data_analysis_report': {},
            'qa_report': {},
            'task': state.get('task', 'Execute data workflow'),
            'next_agent': '',
            'command': 'start'
        }
        
        # Execute data team workflow
        try:
            result = self.data_team_manager.graph.invoke(data_team_state)
            
            # Update CEO state with results
            state['data_team_status'] = True
            state['data_team_report'] = {
                'data_loaded': result.get('data_loaded', False),
                'analysis_done': result.get('analysis_done', False),
                'quality_assurance': result.get('quality_assurance', False),
                'data_analysis_report': result.get('data_analysis_report', {}),
                'qa_report': result.get('qa_report', {})
            }
            state['current_phase'] = 'Data Preparation Complete'
            state['next_phase'] = 'Model Execution'
            
            theme_utility.display_response("Data Team workflow completed successfully!", title=self.agent_name)
            
        except Exception as e:
            theme_utility.display_response(f"Error in Data Team workflow: {str(e)}", title=self.agent_name)
            state['data_team_status'] = False
        
        return Command(
            goto="supervisor",
            update={
                "messages": [chat_utility.build_message_structure(
                    role="assistant", 
                    message="Data Team workflow completed. Ready for next phase."
                )]
            }
        )
    
    def modelling_team_node(self, state: CEOState):
        """Execute Modelling Team workflow"""
        theme_utility.display_response("Executing Modelling Team workflow...", title=self.agent_name)
        
        # Initialize modelling team state
        modelling_team_state = {
            'messages': [{"role": "user", "content": state.get('task', 'Execute modelling workflow')}],
            'task': state.get('task', 'Execute modelling workflow'),
            'meta_model_config': {},
            'model_config': {
                'kpi': ['intercept'],
                'prior_mean': [0],
                'prior_sd': [100],
                'is_random': [1],
                'lower_bound': [float('nan')],
                'upper_bound': [float('nan')],
                'compute_contribution': [0]
            }
        }
        
        # Execute modelling team workflow
        try:
            result = self.modelling_team_manager.graph.invoke(modelling_team_state)
            
            # Update CEO state with results
            state['modelling_team_status'] = True
            state['modelling_team_report'] = {
                'task': result.get('task', ''),
                'meta_model_config': result.get('meta_model_config', {}),
                'model_config': result.get('model_config', {}),
                'final_report': result.get('final_report', '')
            }
            state['current_phase'] = 'Model Execution Complete'
            state['next_phase'] = 'Contribution Analysis'
            
            theme_utility.display_response("Modelling Team workflow completed successfully!", title=self.agent_name)
            
        except Exception as e:
            theme_utility.display_response(f"Error in Modelling Team workflow: {str(e)}", title=self.agent_name)
            state['modelling_team_status'] = False
        
        return Command(
            goto="supervisor",
            update={
                "messages": [chat_utility.build_message_structure(
                    role="assistant", 
                    message="Modelling Team workflow completed. Ready for next phase."
                )]
            }
        )
    
    def contribution_team_node(self, state: CEOState):
        """Execute Contribution Team workflow"""
        theme_utility.display_response("Executing Contribution Team workflow...", title=self.agent_name)
        
        # Initialize contribution team state
        contribution_team_state = {
            'messages': [{"role": "user", "content": state.get('task', 'Execute contribution workflow')}],
            'contribution_analysis_done': False,
            'contribution_interpretation_done': False,
            'contribution_validation_done': False,
            'contribution_analysis_report': {},
            'contribution_interpretation_report': {},
            'contribution_validation_report': {},
            'task': state.get('task', 'Execute contribution workflow'),
            'next_agent': '',
            'command': 'start'
        }
        
        # Execute contribution team workflow
        try:
            result = self.contribution_team_manager.graph.invoke(contribution_team_state)
            
            # Update CEO state with results
            state['contribution_team_status'] = True
            state['contribution_team_report'] = {
                'contribution_analysis_done': result.get('contribution_analysis_done', False),
                'contribution_interpretation_done': result.get('contribution_interpretation_done', False),
                'contribution_validation_done': result.get('contribution_validation_done', False),
                'contribution_analysis_report': result.get('contribution_analysis_report', {}),
                'contribution_interpretation_report': result.get('contribution_interpretation_report', {}),
                'contribution_validation_report': result.get('contribution_validation_report', {})
            }
            state['current_phase'] = 'Contribution Analysis Complete'
            state['next_phase'] = 'Project Complete'
            state['overall_project_status'] = 'Completed'
            
            theme_utility.display_response("Contribution Team workflow completed successfully!", title=self.agent_name)
            
        except Exception as e:
            theme_utility.display_response(f"Error in Contribution Team workflow: {str(e)}", title=self.agent_name)
            state['contribution_team_status'] = False
        
        return Command(
            goto="supervisor",
            update={
                "messages": [chat_utility.build_message_structure(
                    role="assistant", 
                    message="Contribution Team workflow completed. Project phase completed."
                )]
            }
        )
    
    def project_overview_node(self, state: CEOState):
        """Generate project overview"""
        theme_utility.display_response("Generating project overview...", title=self.agent_name)
        
        prompt = CEOAgentPrompt['ceo_overview_prompt'].format(
            data_team_status=state.get('data_team_status', False),
            modelling_team_status=state.get('modelling_team_status', False),
            contribution_team_status=state.get('contribution_team_status', False),
            overall_project_status=state.get('overall_project_status', 'Not Started'),
            current_phase=state.get('current_phase', 'Initialization'),
            next_phase=state.get('next_phase', 'Not Determined'),
            data_team_report=state.get('data_team_report', {}),
            modelling_team_report=state.get('modelling_team_report', {}),
            contribution_team_report=state.get('contribution_team_report', {})
        )
        
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        theme_utility.display_response(response.content, title="Project Overview")
        
        return Command(
            goto="supervisor",
            update={
                "messages": [chat_utility.build_message_structure(
                    role="assistant", 
                    message=f"Project overview generated: {response.content[:100]}..."
                )]
            }
        )
    
    def executive_summary_node(self, state: CEOState):
        """Generate executive summary"""
        theme_utility.display_response("Creating executive summary...", title=self.agent_name)
        
        prompt = CEOAgentPrompt['ceo_executive_summary_prompt'].format(
            data_team_status=state.get('data_team_status', False),
            modelling_team_status=state.get('modelling_team_status', False),
            contribution_team_status=state.get('contribution_team_status', False),
            data_team_report=state.get('data_team_report', {}),
            modelling_team_report=state.get('modelling_team_report', {}),
            contribution_team_report=state.get('contribution_team_report', {})
        )
        
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        state['final_executive_summary'] = response.content
        
        theme_utility.display_response(response.content, title="Executive Summary")
        
        return Command(
            goto="supervisor",
            update={
                "messages": [chat_utility.build_message_structure(
                    role="assistant", 
                    message=f"Executive summary created: {response.content[:100]}..."
                )]
            }
        )
    
    def build_graph(self, state: CEOState):
        """Build the CEO workflow graph"""
        workflow = StateGraph(state)
        
        # Add nodes
        workflow.add_node("supervisor", self.chatNode)
        workflow.add_node("data_team_node", self.data_team_node)
        workflow.add_node("modelling_team_node", self.modelling_team_node)
        workflow.add_node("contribution_team_node", self.contribution_team_node)
        workflow.add_node("project_overview_node", self.project_overview_node)
        workflow.add_node("executive_summary_node", self.executive_summary_node)
        
        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("data_team_node", "supervisor")
        workflow.add_edge("modelling_team_node", "supervisor")
        workflow.add_edge("contribution_team_node", "supervisor")
        workflow.add_edge("project_overview_node", "supervisor")
        workflow.add_edge("executive_summary_node", "supervisor")
        
        return workflow.compile(checkpointer=checkpointer)
    
    def auto_start_workflow(self):
        """Automatically execute the complete workflow"""
        theme_utility.display_response(user_messages['auto_start_mode'], title=self.agent_name)
        
        # Initialize CEO state
        initial_state = {
            'messages': [{"role": "user", "content": "Start complete AutoMMM workflow"}],
            'data_team_status': False,
            'modelling_team_status': False,
            'contribution_team_status': False,
            'overall_project_status': 'In Progress',
            'current_phase': 'Starting Data Team',
            'next_phase': 'Model Execution',
            'data_team_report': {},
            'modelling_team_report': {},
            'contribution_team_report': {},
            'final_executive_summary': '',
            'task': 'Execute complete workflow',
            'next_team': 'data_team',
            'command': 'start'
        }
        
        # Execute the complete workflow
        try:
            result = self.graph.invoke(initial_state)
            theme_utility.display_response("Complete AutoMMM workflow executed successfully!", title=self.agent_name)
            return result
        except Exception as e:
            theme_utility.display_response(f"Error in complete workflow: {str(e)}", title=self.agent_name)
            return None

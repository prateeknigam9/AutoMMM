from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from agent_patterns.states import ContributionTeamManagerState
from utils import chat_utility
from utils import theme_utility
from utils.memory_handler import DataStore

from langchain_ollama import ChatOllama
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from utils.theme_utility import console
from langgraph.types import Command
from langchain_core.prompts.chat import ChatPromptTemplate

from agents.ContributionTeam.ContributionAnalyst import ContributionAnalystAgent
from agents.ContributionTeam.ContributionInterpreter import ContributionInterpreterAgent
from agents.ContributionTeam.ContributionValidator import ContributionValidatorAgent

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

import json


from rich import print

from utils import utility

ManagerPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "contribution_team_manager_prompts",
)

user_messages = utility.load_prompt_config(
    r"prompts\user_messages.yaml",
    "contribution_team_manager",
)


class ContributionTeamManagerAgent:
    def __init__(self, agent_name :str, agent_description:str, backstory:str = ""):
        self.agent_name = f"{agent_name}: Sarah"
        self.agent_description = agent_description
        self.graph = self.build_graph(ContributionTeamManagerState)
        self.backstory = backstory
        self.llm = ChatOllama(model = "llama3.1")
        # self.llm = ChatOpenAI(model = "gpt-4o-mini")
        self.contribution_analyst = ContributionAnalystAgent(
            agent_name="Contribution Analyst Agent",
            agent_description="Analyzes marketing contribution patterns from model outputs, creates contribution reports, and identifies key drivers of marketing performance.",
            model="llama3.1")
        self.contribution_interpreter = ContributionInterpreterAgent(
            agent_name="Contribution Interpreter Agent",
            agent_description="Interprets contribution analysis results, provides business insights, and generates actionable recommendations for marketing optimization.",
            model="llama3.1")
        self.contribution_validator = ContributionValidatorAgent(
            agent_name="Contribution Validator Agent",
            agent_description="Validates contribution analysis results, checks for consistency, and ensures quality of contribution insights and recommendations.",
            model="llama3.1")
        
      
    def chatNode(self, state:ContributionTeamManagerState):
        prompt = ManagerPrompt['manager_chat_prompt'].format(
            agent_name = self.agent_name,
            agent_description = self.agent_description,
            backstory = self.backstory,
            contribution_analysis_status = state['contribution_analysis_done'],
            contribution_interpretation_status = state['contribution_interpretation_done'],
            contribution_validation_status = state['contribution_validation_done']
        )
        
        state['messages'] = state['messages'] + [
                    chat_utility.build_message_structure(role = "system", message = prompt)
                ]
        take_input_prompt = f"Hi, I am {self.agent_name}, How can i help you?"
        while True:            
            user_input = chat_utility.take_user_input(take_input_prompt)
            state['messages'] = state['messages'] + [
                    chat_utility.build_message_structure(role = "user", message = user_input)
                ]
            take_input_prompt = "USER"
            if user_input == "exit":
                break
        
            response = self.llm.invoke(state['messages'])
            theme_utility.display_response(response.content, title = self.agent_name)
            parsed = chat_utility.parse_json_from_response(response.content)
            if parsed and isinstance(parsed, dict) and "call_agent" in parsed:
                if parsed['call_agent'] == "contribution_analyst_agent" and state['contribution_analysis_done'] == True:
                    theme_utility.display_response("contribution analysis is already completed", title = self.agent_name)
                    return Command(
                        goto = "supervisor",
                        update = {
                            "messages": [chat_utility.build_message_structure(role = "assistant", message = "contribution analysis is already completed")]
                            }
                    )
                elif parsed['call_agent'] == "contribution_interpreter_agent" and state['contribution_interpretation_done'] == True:
                    theme_utility.display_response("contribution interpretation is already completed", title = self.agent_name)
                    return Command(
                        goto = "supervisor",
                        update = {
                            "messages": [chat_utility.build_message_structure(role = "assistant", message = "contribution interpretation is already completed")]
                            }
                    )
                elif parsed['call_agent'] == "contribution_validator_agent" and state['contribution_validation_done'] == True:
                    theme_utility.display_response("contribution validation is already completed", title = self.agent_name)
                    return Command(
                        goto = "supervisor",
                        update = {
                            "messages": [chat_utility.build_message_structure(role = "assistant", message = "contribution validation is already completed")]
                            }
                    )
                else:
                    approved, feedback = chat_utility.ask_user_approval(agent_name = parsed['call_agent']) 
                    if approved is True:
                        response = f"running {parsed['call_agent']} with input {parsed.get("task", "")}"
                        messages = state['messages'] + [
                            chat_utility.build_message_structure(role = "assistant", message = response)
                        ]
                        return Command(
                            goto = parsed['call_agent'],
                            update = {
                                "next_agent": parsed['call_agent'],
                                "task": parsed.get("task", ""),
                                "messages" : messages
                                }
                        )
                    else:
                        response = f"user denied to move to {parsed['call_agent']}, with this input : {feedback}"
                        chat_utility.append_to_structure(state['messages'], role="assistant", message = response)
            else:
                chat_utility.append_to_structure(state['messages'], role="assistant", message = response.content)

    
    def contribution_analyst_node(self, state: ContributionTeamManagerState):
        theme_utility.print_startup_info(
            agent_name=self.contribution_analyst.agent_name,
            agent_description=self.contribution_analyst.agent_description,
            is_interactive=False,
        )

        inputs = {
            "messages": [
                chat_utility.build_message_structure(role="user", message=state['task'])
            ]
        }
        ca_response = self.contribution_analyst.graph.invoke(inputs)
        if ca_response['completed']:
            state['contribution_analysis_done'] = True
    
        messages_from_contribution_analyst = ca_response['messages']
        sysprompt = f"You are {self.contribution_analyst.agent_name},{self.contribution_analyst.agent_description} , working for a {self.agent_name}, based on you history of messages given by user, reply back to him on completion or status of your allotted task."
        messages = [
            chat_utility.build_message_structure(role = "system", message = sysprompt),
            chat_utility.build_message_structure(role = "user", message = json.dumps(messages_from_contribution_analyst))
        ]
        response = self.llm.invoke(messages)
        theme_utility.display_response(response.content, title = self.contribution_analyst.agent_name)
        approved, feedback = chat_utility.ask_user_approval(agent_name = self.contribution_analyst.agent_name, prompt_suffix="Use this input, or modify before sending to manager agent?") 
        if approved is True:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": response.content,
                    "messages": [chat_utility.build_message_structure(role = "assistant", message = response.content)],
                    'contribution_analysis_done': True
                    }
            )
        else:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": feedback,
                    "messages": [
                        chat_utility.build_message_structure(role = "assistant", message = response.content),
                        chat_utility.build_message_structure(role = "user", message = feedback)
                        ],
                    'contribution_analysis_done': True
                    }
            )
    

    def contribution_interpreter_node(self, state: ContributionTeamManagerState):
        theme_utility.print_startup_info(
            agent_name=self.contribution_interpreter.agent_name,
            agent_description=self.contribution_interpreter.agent_description,
            is_interactive=False,
        )
        inputs = {
            "messages": [
                chat_utility.build_message_structure(role="user", message=state['task'])
            ],
            "completed" :None
        }
        ci_response = self.contribution_interpreter.graph.invoke(inputs)
        if ci_response['completed']:
            state['contribution_interpretation_done'] = True
            state['contribution_interpretation_report'] = {
                'business_insights': ci_response['business_insights'],
                'actionable_recommendations': ci_response['actionable_recommendations'],
                'marketing_optimization': ci_response['marketing_optimization'] 
                }     
        messages_from_contribution_interpreter = ci_response['messages']
        sysprompt = f"You are {self.contribution_interpreter.agent_name},{self.contribution_interpreter.agent_description} working for a {self.agent_name}, based on you history of messages and interpretation_report given by user, reply back to him on the status of your allotted task with a short summary"
        messages = [
            chat_utility.build_message_structure(role = "system", message = sysprompt),
            chat_utility.build_message_structure(role = "user", message = json.dumps(messages_from_contribution_interpreter)),
            chat_utility.build_message_structure(role = "user", message = json.dumps(state['contribution_interpretation_report'])),
        ]
        response = self.llm.invoke(messages)
        theme_utility.display_response(response.content, title = self.contribution_interpreter.agent_name)
        with console.status("[plum1] Generating and saving contribution context...", spinner="dots"):
            contribution_context = self.generate_contribution_context("memory/")
        approved, feedback = chat_utility.ask_user_approval(agent_name = self.contribution_interpreter.agent_name, prompt_suffix="Use this input, or modify before sending to manager agent?") 
        if approved is True:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": response.content,
                    "messages": [chat_utility.build_message_structure(role = "assistant", message = response.content)],
                    "contribution_interpretation_done": True
                    }
            )
        else:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": feedback,
                    "messages": [
                        chat_utility.build_message_structure(role = "assistant", message = response.content),
                        chat_utility.build_message_structure(role = "user", message = feedback)
                        ],
                    "contribution_interpretation_done": True
                    }
            )
    
    def contribution_validator_node(self, state: ContributionTeamManagerState):
        theme_utility.print_startup_info(
            agent_name=self.contribution_validator.agent_name,
            agent_description=self.contribution_validator.agent_description,
            is_interactive=False,
        )
        inputs = {
            "messages": [
                chat_utility.build_message_structure(role="user", message=state['task'])
            ],
            "completed" :None
        }
        cv_response = self.contribution_validator.graph.invoke(inputs)
        if cv_response['completed']:
            state['contribution_validation_done'] = True
            state['contribution_validation_report'] = {
                'validation_report': cv_response['validation_report'],
                'validation_report_path': cv_response['validation_report_path']
                }           
        messages_from_contribution_validator = cv_response['messages']
        sysprompt = f"You are {self.contribution_validator.agent_name},{self.contribution_validator.agent_description} working for a {self.agent_name}, based on you history of messages and validation_report given by user, reply back to him on the status of your allotted task with a short summary"
        messages = [
            chat_utility.build_message_structure(role = "system", message = sysprompt),
            chat_utility.build_message_structure(role = "user", message = json.dumps(messages_from_contribution_validator)),
            chat_utility.build_message_structure(role = "user", message = json.dumps(state['contribution_interpretation_report'])),
        ]
        response = self.llm.invoke(messages)
        theme_utility.display_response(response.content, title = self.contribution_validator.agent_name)
        approved, feedback = chat_utility.ask_user_approval(agent_name = self.contribution_validator.agent_name, prompt_suffix="Use this input, or modify before sending to manager agent?") 
        if approved is True:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": response.content,
                    "messages": [chat_utility.build_message_structure(role = "assistant", message = response.content)],
                    "contribution_validation_done":True
                    }
            )
        else:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": feedback,
                    "messages": [
                        chat_utility.build_message_structure(role = "assistant", message = response.content),
                        chat_utility.build_message_structure(role = "user", message = feedback)
                        ],
                    "contribution_validation_done":True
                    }
            )

    def generate_contribution_context(self, memory_dir: str):
        context = []
        for file_path in Path(memory_dir).glob("*.txt"):
            file_name = file_path.name.split('.')[0]
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    content = json.load(f)
                    content_text = json.dumps(content, indent=2)
                except json.JSONDecodeError:
                    f.seek(0)  
                    content_text = f.read()
                context.append(f"### {file_name}\n{content_text.strip()}")

        contribution_context = "\n\n".join(context)
        system_prompt = user_messages['generate_contribution_context']
        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (contribution_context)}
        ])
        DataStore.set_str("contribution_context", response.content.strip())
        utility.save_to_memory_file("contribution_context.txt", response.content.strip())
        return response.content.strip()

    def build_graph(self, state:ContributionTeamManagerState):
        workflow = StateGraph(state)
        workflow.add_node("supervisor", self.chatNode)
        workflow.add_node("contribution_analyst_agent", self.contribution_analyst_node)
        workflow.add_node("contribution_interpreter_agent", self.contribution_interpreter_node)
        workflow.add_node("contribution_validator_agent", self.contribution_validator_node)

        workflow.add_edge(START, "supervisor")
        # workflow.add_edge("supervisor", END)

        return workflow.compile(checkpointer= checkpointer)

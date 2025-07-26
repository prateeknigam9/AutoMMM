from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from agent_patterns.states import DataTeamManagerState
from utils import chat_utility
from utils import theme_utility

from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langgraph.types import Command
from langchain_core.prompts.chat import ChatPromptTemplate

from agents.DataHandlingTeam.DataEngineer import DataEngineerAgent
from agents.DataHandlingTeam.DataAnalyst import DataAnalystAgent
from agents.DataHandlingTeam.DataQualityAnalyst import DataQualityAnalystAgent

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

import json


from rich import print

from utils import utility

ManagerPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "supervisor_router",
)


class DataTeamManagerAgent:
    def __init__(self, agent_name :str, agent_description:str, backstory:str = ""):
        self.agent_name = f"{agent_name}: Gaurav"
        self.agent_description = agent_description
        self.graph = self.build_graph(DataTeamManagerState)
        self.backstory = backstory
        self.llm = ChatOllama(model = "llama3.1")
        # self.llm = ChatOpenAI(model = "gpt-4o-mini")
        self.data_engineer = DataEngineerAgent(
            agent_name="Data Engineer",
            agent_description="Responsible for loading, and contextualizing raw MMM input data using Excel files and a UI-based interaction, and preparing it for downstream modeling.",
            model="llama3.1")
        self.data_analyst = DataAnalystAgent(
            agent_name="Data Analyst",
            agent_description="Performs data profiling, summarization, and column categorization to ensure structured, ready-to-model datasets.",
            model="llama3.1")
        self.qa_analyst = DataQualityAnalystAgent(
            agent_name="Data Quality Specialist Agent",
            agent_description="Validates brand-level and product-level data for modeling readiness by running automated checks, summarizing tool outputs, and generating a structured validation report.",
            model="llama3.1")
      
    def chatNode(self, state:DataTeamManagerState):
        prompt = ("you are {agent_name}, {agent_description}, reply to the user in a conversational manner, Backstory : {backstory},"
                  """ Current Status of agents already called:
                        data_engineer_agent: {data_engineer_status}
                        data_analysis_agent: {data_analyst_status}
                        quality_assurance_agent : {qa_analyst_status}
                        """
                  """ if the user intent requires calling agent, only then Reply in json format
                            - `call_agent` (str) : (if any) the next agent to handle the task Literal["data_engineer_agent", "data_analysis_agent","quality_assurance_agent","__end__"]
                            - `task` (str) : a brief task description for the agent
                  """).format(
            agent_name = self.agent_name,
            agent_description = self.agent_description,
            backstory = self.backstory,
            data_engineer_status = state['data_loaded'],
            data_analyst_status = state['analysis_done'],
            qa_analyst_status = state['quality_assurance']
        )
        take_input_prompt = f"Hi, I am {self.agent_name}, How can i help you?"

        while True:            
            if state["task"] is not None and state['next_agent'] == "supervisor":
                    user_input = state["task"]
                    state["task"] = None
            else:
                user_input = chat_utility.take_user_input(take_input_prompt)
            take_input_prompt = "USER:"
            if user_input == "exit":
                break

            messages = [
                chat_utility.build_message_structure(role = "system", message = prompt),
                chat_utility.build_message_structure(role = "user", message = user_input),
            ]
            response = self.llm.invoke(messages)
            theme_utility.display_response(response.content, title = self.agent_name)
            parsed = chat_utility.parse_json_from_response(response.content)
            if parsed and isinstance(parsed, dict) and "call_agent" in parsed:
                if parsed['call_agent'] == "data_engineer_agent" and state['data_loaded'] == True:
                    return Command(
                        goto = "supervisor",
                        update = {
                            "task": "data is already loaded"
                            }
                    )
                    break
                elif parsed['call_agent'] == "data_analyst_agent" and state['analysis_done'] == True:
                    return Command(
                        goto = "supervisor",
                        update = {
                            "task": "data analysis is already done and saved"
                            }
                    )
                    break
                elif parsed['call_agent'] == "quality_assurance_agent" and state['quality_assurance'] == True:
                    return Command(
                        goto = "supervisor",
                        update = {
                            "task": "quality assurance is done"
                            }
                    )
                    break
                else:
                    approved, feedback = chat_utility.ask_user_approval(agent_name = parsed['call_agent']) 
                    if approved is True:
                        return Command(
                            goto = parsed['call_agent'],
                            update = {
                                "next_agent": parsed['call_agent'],
                                "task": parsed.get("task", "")
                                }
                        )
                        break
                    else:
                        return Command(
                            goto = "supervisor",
                            update = {
                                "next_agent": "supervisor",
                                "task": f"user denied to move to {parsed['call_agent']}, with this input : {feedback}"
                                }
                        )
                        break
    
    def data_engineer_node(self, state: DataTeamManagerState):
        theme_utility.print_startup_info(
            agent_name=self.data_engineer.agent_name,
            agent_description=self.data_engineer.agent_description,
            is_interactive=False,
        )
        inputs = {
            "messages": [
                chat_utility.build_message_structure(role="user", message=state['task'])
            ]
        }
        de_response = self.data_engineer.graph.invoke(inputs)
        if de_response['completed']:
            state['data_loaded'] == True
        messages_from_data_engineer = de_response['messages']
        sysprompt = f"You are {self.data_engineer.agent_name},{self.data_engineer.agent_description} , working for a {self.agent_name}, based on you history of messages given by user, reply back to him on completion or status of your allotted task."
        messages = [
            chat_utility.build_message_structure(role = "system", message = sysprompt),
            chat_utility.build_message_structure(role = "user", message = json.dumps(messages_from_data_engineer))
        ]
        response = self.llm.invoke(messages)
        theme_utility.display_response(response.content, title = self.data_engineer.agent_name)
        approved, feedback = chat_utility.ask_user_approval(agent_name = self.data_engineer.agent_name, prompt_suffix="Use this input, or modify before sending to manager agent?") 
        if approved is True:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": response.content
                    }
            )
        else:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": feedback
                    }
            )

    def data_analyst_node(self, state: DataTeamManagerState):
        theme_utility.print_startup_info(
            agent_name=self.data_analyst.agent_name,
            agent_description=self.data_analyst.agent_description,
            is_interactive=False,
        )
        inputs = {
            "messages": [
                chat_utility.build_message_structure(role="user", message=state['task'])
            ],
            "completed" :None
        }
        da_response = self.data_analyst.graph.invoke(inputs)
        if da_response['completed']:
            state['analysis_done'] == True
            state['data_analysis_report'] = {
                'data_summary': da_response['data_summary'],
                'column_categories': da_response['column_categories'],
                'distinct_products': da_response['distinct_products'] 
                }           
        messages_from_data_analyst = da_response['messages']
        sysprompt = f"You are {self.data_analyst.agent_name},{self.data_analyst.agent_description} working for a {self.agent_name}, based on you history of messages and data_report given by user, reply back to him on the status of your allotted task with a short summary"
        messages = [
            chat_utility.build_message_structure(role = "system", message = sysprompt),
            chat_utility.build_message_structure(role = "user", message = json.dumps(messages_from_data_analyst)),
            chat_utility.build_message_structure(role = "user", message = json.dumps(state['data_analysis_report'])),
        ]
        response = self.llm.invoke(messages)
        theme_utility.display_response(response.content, title = self.data_analyst.agent_name)
        approved, feedback = chat_utility.ask_user_approval(agent_name = self.data_analyst.agent_name, prompt_suffix="Use this input, or modify before sending to manager agent?") 
        if approved is True:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": response.content
                    }
            )
        else:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": feedback
                    }
            )
    
    def data_qa_analyst_node(self, state: DataTeamManagerState):
        theme_utility.print_startup_info(
            agent_name=self.qa_analyst.agent_name,
            agent_description=self.qa_analyst.agent_description,
            is_interactive=False,
        )
        inputs = {
            "messages": [
                chat_utility.build_message_structure(role="user", message=state['task'])
            ],
            "completed" :None
        }
        qas_response = self.qa_analyst.graph.invoke(inputs)
        if qas_response['completed']:
            state['analysis_done'] == True
            state['qa_report'] = {
                'qa_analyst_report': qas_response['final_report'],
                'qa_analyst_report_path': qas_response['report_path']
                }           
        messages_from_data_quality_specialist = qas_response['messages']
        sysprompt = f"You are {self.qa_analyst.agent_name},{self.qa_analyst.agent_description} working for a {self.agent_name}, based on you history of messages and qa_report given by user, reply back to him on the status of your allotted task with a short summary"
        messages = [
            chat_utility.build_message_structure(role = "system", message = sysprompt),
            chat_utility.build_message_structure(role = "user", message = json.dumps(messages_from_data_quality_specialist)),
            chat_utility.build_message_structure(role = "user", message = json.dumps(state['data_analysis_report'])),
        ]
        response = self.llm.invoke(messages)
        theme_utility.display_response(response.content, title = self.qa_analyst.agent_name)
        approved, feedback = chat_utility.ask_user_approval(agent_name = self.qa_analyst.agent_name, prompt_suffix="Use this input, or modify before sending to manager agent?") 
        if approved is True:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": response.content
                    }
            )
        else:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": feedback
                    }
            )

    

    def build_graph(self, state:DataTeamManagerState):
        workflow = StateGraph(state)
        workflow.add_node("supervisor", self.chatNode)
        workflow.add_node("data_engineer_agent", self.data_engineer_node)
        workflow.add_node("data_analysis_agent", self.data_analyst_node)
        workflow.add_node("quality_assurance_agent", self.data_qa_analyst_node)

        workflow.add_edge(START, "supervisor")
        # workflow.add_edge("supervisor", END)

        return workflow.compile(checkpointer= checkpointer)
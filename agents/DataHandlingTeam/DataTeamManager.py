from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from agent_patterns.states import DataTeamManagerState
from utils import chat_utility
from utils import theme_utility
from utils.memory_handler import DataStore

from langchain_ollama import ChatOllama
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from utils.theme_utility import console
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
    "data_team_manager_prompts",
)

user_messages = utility.load_prompt_config(
    r"prompts\user_messages.yaml",
    "data_team_manager",
)


class DataTeamManagerAgent:
    def __init__(self, agent_name :str, agent_description:str, backstory:str = ""):
        self.agent_name = f"{agent_name}: Vibhor"
        self.agent_description = agent_description
        self.graph = self.build_graph(DataTeamManagerState)
        self.backstory = backstory
        self.llm = ChatOllama(model = "llama3.1")
        # self.llm = ChatOpenAI(model = "gpt-4o-mini")
        self.data_engineer = DataEngineerAgent(
            agent_name="Data Engineer Agent",
            agent_description="Responsible for loading, and contextualizing raw MMM input data using Excel files and a UI-based interaction, and preparing it for downstream modeling.",
            model="llama3.1")
        self.data_analyst = DataAnalystAgent(
            agent_name="Data Analyst Agent",
            agent_description="Performs data profiling, summarization, and column categorization to ensure structured, ready-to-model datasets.",
            model="llama3.1")
        self.qa_analyst = DataQualityAnalystAgent(
            agent_name="Data Quality Specialist Agent",
            agent_description="Validates brand-level and product-level data for modeling readiness by running automated checks, summarizing tool outputs, and generating a structured validation report.",
            model="llama3.1")
        
      
    def chatNode(self, state:DataTeamManagerState):
        prompt = ManagerPrompt['manager_chat_prompt'].format(
            agent_name = self.agent_name,
            agent_description = self.agent_description,
            backstory = self.backstory,
            data_engineer_status = state['data_loaded'],
            data_analyst_status = state['analysis_done'],
            qa_analyst_status = state['quality_assurance']
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
                if parsed['call_agent'] == "data_engineer_agent" and state['data_loaded'] == True:
                    theme_utility.display_response("data is already loaded", title = self.agent_name)
                    return Command(
                        goto = "supervisor",
                        update = {
                            "messages": [chat_utility.build_message_structure(role = "assistant", message = "data is already loaded")]
                            }
                    )
                elif parsed['call_agent'] == "data_analyst_agent" and state['analysis_done'] == True:
                    theme_utility.display_response("data analysis is already done and saved", title = self.agent_name)
                    return Command(
                        goto = "supervisor",
                        update = {
                            "messages": [chat_utility.build_message_structure(role = "assistant", message = "data analysis is already done and saved")]
                            }
                    )
                elif parsed['call_agent'] == "quality_assurance_agent" and state['quality_assurance'] == True:
                    theme_utility.display_response("quality assurance is done", title = self.agent_name)
                    return Command(
                        goto = "supervisor",
                        update = {
                            "messages": [chat_utility.build_message_structure(role = "assistant", message = "quality assurance is done")]
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
            state['data_loaded'] = True
    
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
                    "task": response.content,
                    "messages": [chat_utility.build_message_structure(role = "assistant", message = response.content)],
                    'data_loaded': True
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
                    'data_loaded': True
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
            state['analysis_done'] = True
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
        with console.status("[plum1] Generating and saving memory context...", spinner="dots"):
            memory_context = self.generate_memory_context("memory/")
        approved, feedback = chat_utility.ask_user_approval(agent_name = self.data_analyst.agent_name, prompt_suffix="Use this input, or modify before sending to manager agent?") 
        if approved is True:
            return Command(
                goto = "supervisor",
                update = {
                    "next_agent": "supervisor",
                    "task": response.content,
                    "messages": [chat_utility.build_message_structure(role = "assistant", message = response.content)],
                    "analysis_done": True
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
                    "analysis_done": True
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
                'qa_analyst_report': qas_response['qa_report'],
                'qa_analyst_report_path': qas_response['qa_report_path']
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
                    "task": response.content,
                    "messages": [chat_utility.build_message_structure(role = "assistant", message = response.content)],
                    "quality_assurance":True
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
                    "quality_assurance":True
                    }
            )

    def generate_memory_context(self, memory_dir: str):
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

        memory_context = "\n\n".join(context)
        system_prompt = user_messages['generate_memory_context']
        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (memory_context)}
        ])
        DataStore.set_str("memory_context", response.content.strip())
        utility.save_to_memory_file("memory_context.txt", response.content.strip())
        return response.content.strip()

    def build_graph(self, state:DataTeamManagerState):
        workflow = StateGraph(state)
        workflow.add_node("supervisor", self.chatNode)
        workflow.add_node("data_engineer_agent", self.data_engineer_node)
        workflow.add_node("data_analysis_agent", self.data_analyst_node)
        workflow.add_node("quality_assurance_agent", self.data_qa_analyst_node)

        workflow.add_edge(START, "supervisor")
        # workflow.add_edge("supervisor", END)

        return workflow.compile(checkpointer= checkpointer)
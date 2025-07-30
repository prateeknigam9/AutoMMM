from binascii import Incomplete
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig


from agent_patterns.react_agent.react_agent import ReactAgentState
from utils import utility
from utils import chat_utility
from utils import theme_utility
from utils.theme_utility import console, log

from typing import Literal, TypedDict, Annotated, List, Optional
from langchain_core.messages import AnyMessage
import operator
import json
from pydantic import BaseModel, Field

from rich import print

ReActPrompts = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "newReActAgentPrompt",
)




class TaskToolPair(BaseModel):
    task: str
    suggested_tool: str
    reason: str
    sequence_id: int
    dependency : list[int]

class ToolAnalysisResponse(BaseModel):
    user_query: str
    task_tool_pairs: List[TaskToolPair]
    conversational_reply: str


class ReactAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    query: str
    task_tool_pairs : List[TaskToolPair]
    task_tool_pair_responses: dict

class ReactAgent:
    def __init__(self, agent_name: str, tools:list=[], backstory:str="",provider :Literal["ollama", "groq","openai"] = "ollama", language_model:str = "llama3.1", tool_running_model :str = "llama3-groq-tool-use:8b",):
        self.agent_name = agent_name
        self.tools = tools
        self.graph = self.build_graph(ReactAgentState)
        self.tool_dict = {tool.name: tool for tool in tools}        
        self.tools_list = ", ".join([f"`{tool.name}`: {tool.description}" for tool in self.tools])
        if provider == "groq":            
            self.llm = ChatGroq(model = "qwen/qwen3-32b")
            self.llm_with_tools = ChatGroq(model = "qwen/qwen3-32b").bind_tools(tools)
        elif provider == "ollama":
            self.llm = ChatOllama(model = language_model)
            self.llm_with_tools = ChatOllama(model = tool_running_model).bind_tools(tools)
        elif provider == "openai":
            self.llm_with_tools = ChatOpenAI(model = "gpt-4o-mini").bind_tools(tools)
            self.llm = ChatOpenAI(model = "gpt-4o-mini")
    
    # step 1 - break the query into task - tool pair
    def queryBreakerNode(self, state: ReactAgentState):
        log("[medium_purple3]LOG: Entered queryBreakerNode[/]")
        with console.status(f"[plum1] queryBreakerNode setting up...[/]\n", spinner="dots"):  
            state['query'] = state['messages'][0].content 
            prompt = ReActPrompts["queryBreakerPrompt"].format(
                role = self.agent_name,
                query = state['messages'][0].content,
                tool_list = utility.tools_to_action_prompt(self.tools)
            )            
            structured_resp_llm = self.llm.with_structured_output(ToolAnalysisResponse)
            response = structured_resp_llm.invoke(prompt)
            theme_utility.display_task_list(response.task_tool_pairs)
            theme_utility.display_response(response.conversational_reply)
        return {
            "task_tool_pairs": response.task_tool_pairs,
            "messages": [response.conversational_reply]
            }
    
    
    def task_running_agent(self, state: ReactAgentState, config: RunnableConfig):
        log("[medium_purple3]LOG: Entered queryBreakerNode[/]")
        with console.status(f"[plum1] queryBreakerNode setting up...[/]\n", spinner="dots"):  
            sysprompt = [
                chat_utility.build_message_structure(
                    role="system",
                    message="You are a helpful AI assistant, please respond to the users query to the best of your ability!, use previous responses as reference if required"
                )
            ]
            previous_responses = {}
            task_tool_pair_responses = []
            tries = 0
            completed_task_ids = set()
        while tries < 3:
            incomplete_tasks = [tasks for tasks in state["task_tool_pairs"] if tasks.sequence_id not in completed_task_ids]
            if not incomplete_tasks:
                break
            for tasks in incomplete_tasks:
                llm_with_tools = ChatOllama(model = "llama3-groq-tool-use:8b").bind_tools([self.tool_dict[tasks.suggested_tool]])

                previous_responses_str = "\n".join([f"{key}:{value}" for key, value in previous_responses.items()])
                user_msg = [
                    chat_utility.build_message_structure(
                        role = "user",
                        message = f"\nTASK:{tasks.task}\nPREVIOUS RESPONSES: {previous_responses_str}"
                    )
                ]
                response = llm_with_tools.invoke(
                            sysprompt + user_msg,
                            config
                            )
                print(sysprompt + user_msg)
                print(response)
                
                if response.tool_calls:                
                    tool_results = []
                    for tool_call in response.tool_calls:
                        print("NAME",tool_call["name"])
                        print("ARGS",tool_call["args"])
                        tool_result = self.tool_dict[tool_call["name"]].invoke(tool_call["args"])

                        tool_results.append(json.dumps(tool_result))
                        print("tool_results:", tool_results)
                    previous_responses[tasks.task] = tool_results
                    task_tool_pair_responses.append({
                                "task" : tasks.task,
                                "result" : tool_results
                    })
                    completed_task_ids.add(tasks.sequence_id)
                    print("=============================")
            tries += 1
        print(task_tool_pair_responses)
        return {"task_tool_pair_responses": task_tool_pair_responses}

    def respond_node(self, state: ReactAgentState):
        prompt = (
            f"Based on the below query and observations provided by the assistant in a friendly manner, reply to the user."
            f"\nObservations: {state['task_tool_pair_responses']}"
            f"\nQuery: {state['query']}"
        )
        response = ChatOllama(model = "llama3.1").invoke(prompt).content
        theme_utility.display_response(response)
        return {"messages":  [response]}

    def build_graph(self,state: ReactAgentState):
        workflow = StateGraph(state)
        workflow.add_node("queryBreakerNode",self.queryBreakerNode)
        workflow.add_node("task_running_agent",self.task_running_agent)
        workflow.add_node("respond_node",self.respond_node)

        workflow.add_edge(START, "queryBreakerNode")
        workflow.add_edge("queryBreakerNode", "task_running_agent")
        workflow.add_edge("task_running_agent", "respond_node")
        workflow.add_edge("respond_node", END)
        

        return workflow.compile(name = self.agent_name)
    
    def run(self, query):
        state = {
            "messages" : [chat_utility.build_message_structure(role = "user",message=query)],
            "query": query,
            "task_tool_pair": [],
            "task_tool_pair_responses": {}
        }
        result = self.graph.invoke(state)
        print("================================")
        print(result)


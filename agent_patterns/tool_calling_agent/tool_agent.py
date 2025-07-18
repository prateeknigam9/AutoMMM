from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command

from utils import utility
from utils import chat_utility
from utils import theme_utility
from utils.theme_utility import console, log, SingleSpinner
from agent_patterns.states import ToolAgentState
import ast
import operator
import json
from typing import Annotated, TypedDict, List, Literal, Any
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing import Union

ToolPrompts = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "ToolAgent",
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

class argGenResposne(BaseModel):
    tool: str
    tool_args: dict

class ToolAgent:
    def __init__(self, agent_name:str, agent_goal:str, agent_description:str, tools:list, model:str, log_path:str = "tool_agent.log", async_mode = False):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.agent_goal = agent_goal

        self.tools = tools
        self.tool_desc, self.tool_names = utility.tools_to_action_prompt(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        self.llm = ChatOllama(model=model)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.async_mode = async_mode

        self.graph = self.build_graph()

    # step 1 - break the query into task - tool pair
    def queryBreakerNode(self, state: ToolAgentState):
        log("[medium_purple3]LOG: Entered queryBreakerNode[/]")
        with console.status(f"[plum1] queryBreakerNode setting up...[/]\n", spinner="dots"):  
            prompt = ToolPrompts["queryBreakerPrompt"].format(
                role = self.agent_name,
                query=state["query"],
                tool_list=utility.tools_to_action_prompt(self.tools),
            )
            if len(state["user_suggestions"]) > 0:
                prompt += "SUGGESTION: " + ", ".join(state["user_suggestions"])
            
            structured_resp_llm = self.llm.with_structured_output(ToolAnalysisResponse)
            response = structured_resp_llm.invoke(prompt)
            theme_utility.display_task_list(response.task_tool_pairs)
            theme_utility.display_response(response.conversational_reply)
            log("[medium_purple3]LOG: queryBreakerNode generated response[/]")
        approved, suggestion = chat_utility.take_user_feedback("query Breaker")
        if approved is True:
            return Command(
                goto="taskRunnerNode",
                update = {
                    "task_tool_pairs": response.task_tool_pairs,
                    "message": [response.conversational_reply],
                }
            )
        else:
            log("[medium_purple3]LOG: User requested modifications; executing with updated suggestions.[/]")
            return Command(
                goto="queryBreakerNode",
                update={"user_suggestions": suggestion})

    # step 2 - tool runner and storing results
    def taskRunnerNode(self, state: ToolAgentState):
        log("[medium_purple3]LOG: Entered taskRunnerNode[/]")
        with console.status(f"[plum1] taskRunnerNode setting up...[/]\n", spinner="dots"):
            tool_resp_dict = []
            task_results = {}
            sorted_tasks = sorted(state["task_tool_pairs"], key=lambda x: x.sequence_id)

        for task_tool in sorted_tasks:
            dependent_outputs = (
                {dep_id: task_results[dep_id] for dep_id in task_tool.dependency}
                if task_tool.dependency else {}
            )
            arg_prompt = ToolPrompts['argGeneratorPrompt'].format(
                tool = task_tool.suggested_tool,
                task = task_tool.task,
                prev_result = dependent_outputs,
                tool_desc = utility.tools_to_action_prompt([self.tools_by_name[task_tool.suggested_tool]])
            )
            tool_args_str = self.llm.with_structured_output(argGenResposne).invoke(arg_prompt)
            # tool_args = ast.literal_eval(tool_args_str.tool_args)
            print(type(tool_args_str.tool_args))
            tool_args = tool_args_str.tool_args
            log(f"[medium_purple3]LOG: Running tool for task={task_tool.task}[/]")
            approved, suggestion = chat_utility.ask_user_approval(
                agent_name="taskRunnerNode",
                prompt_suffix=chat_utility.tool_approval_msg(
                    tool=task_tool.suggested_tool,
                    task=task_tool.task,
                    reason=task_tool.reason,
                    args=tool_args,
                ),
            )
            if approved is True:
                result = self.tools_by_name[task_tool.suggested_tool].invoke(
                    tool_args
                )
                tool_inputs = dict(task_tool)
                tool_result = {
                        "task": task_tool.task,
                        "tool": task_tool.suggested_tool,
                        "args": tool_args,
                        "result": result,
                    }
                tool_resp_dict.append(tool_result)
                task_results[task_tool.sequence_id] = result
                chat_utility.append_to_structure(history = state['message'], role = "assistant", message = json.dumps(tool_inputs))
                chat_utility.append_to_structure(history = state['message'], role = "tool", message = result)
            else:
                log(f"[medium_purple3]LOG: User provided modification suggestion: {suggestion}[/]")
                while True:
                    try:
                        parsed = ast.literal_eval(suggestion)
                        if isinstance(parsed, dict):
                            result = self.tools_by_name[parsed["tool_name"]].invoke(
                                parsed["tool_args"]
                            )
                            tool_input = {
                                    "task": task_tool.task,
                                    "tool": parsed["tool_name"],
                                    "args": parsed["tool_args"]
                                }
                            tool_result = {
                                    "task": task_tool.task,
                                    "tool": parsed["tool_name"],
                                    "args": parsed["tool_args"],
                                    "result": result,
                                }
                            tool_resp_dict.append(tool_result)
                            task_results[task_tool.sequence_id] = result
                            chat_utility.append_to_structure(history = state['message'], role = "assistant", message = json.dumps(tool_inputs))
                            chat_utility.append_to_structure(history = state['message'], role = "tool", message = result)
                            break
                    except:
                        log("[medium_purple3]LOG: Failed to parse user suggestion[/]")
                        _, suggestion = chat_utility.ask_user_approval(
                            agent_name="Category Approval Node",
                            prompt_suffix="Invalid input. Please respond with a dictionary like:\n"
                            "{'tool_name': 'tool_x', 'tool_args': {...}}\n"
                            "Or type 'skip' to deny this tool.",
                        )
                        if suggestion == "skip":
                            log("[medium_purple3]LOG: User skipped the tool[/]")
                            tool_input = {
                                    "task": task_tool.task,
                                    "tool": None,
                                    "args": None,
                                }
                            tool_result = {
                                    "task": task_tool.task,
                                    "tool": None,
                                    "args": None,
                                    "result": "denied",
                                }
                            tool_resp_dict.append(tool_result)
                            chat_utility.append_to_structure(history = state['message'], role = "assistant", message = json.dumps(tool_inputs))
                            chat_utility.append_to_structure(history = state['message'], role = "tool", message = "denied")
                            break
            task_results[task_tool.sequence_id] = result
        with console.status(f"[plum1] Generating response...[/]\n", spinner="dots"):
            prompt = ToolPrompts['ToolNodePrompt']
            prompt += f"\n QUERY - {state['query']}"
            prompt += f"\n TOOL RESULTS - {tool_resp_dict}"            
            response = self.llm.invoke(prompt.strip())
        theme_utility.display_response(response.content)
        log(f"[medium_purple3]LOG: Generated tool response[/]")
        user_approval, user_suggestion = chat_utility.take_user_feedback("taskRunnerNode")
        if user_approval is True:
            return Command(
                goto="responseNode",
                update = {"message": [response.content], "tool_response_list": tool_resp_dict}
            )
        else:
            log(f"[medium_purple3]LOG: User did not approve final output, rerouting to queryBreakerNode...[/]")
            return Command(
                goto="queryBreakerNode",
                update={"message": [response.content], "user_suggestions": [user_suggestion]},
            )

    # step 3 - explaining the tool response
    def responseNode(self, state: ToolAgentState):
        log("[medium_purple3]LOG: Entered finalResponseNode[/]")
        with console.status(f"[plum1] Generating response...[/]\n", spinner="dots"):
            prompt = ToolPrompts["ResponseNodePrompt"].format(
                role = self.agent_name,
                goal = self.agent_goal,
                description = self.agent_description
            )
            prompt += f"\nQUERY : \n{state['query']}"
            prompt += f"\nTOOL RESULTS : \n{state['tool_response_list']}"
            response = self.llm.invoke(prompt.strip())
        theme_utility.display_response(response.content)
        log("[medium_purple3]LOG: Final response generated[/]")
        user_approval, user_suggestion = chat_utility.take_user_feedback(
            "responseNode",
            prompt_suffix=(
                "Do you approve this response?\n"
                "If not, please specify which node to re-run: taskRunnerNode or queryBreakerNode."
            )
        )
        if user_approval is True:
            return Command(
                goto = END,
                update = {
                    "message": [response.content],
                    "finalresponse": response.content,
                }
            )
        while True:
            try:
                log(f"[medium_purple3]LOG: User did not approve final output, rerouting to {user_suggestion}...[/]")
                return Command(
                    goto=user_suggestion,
                    update={
                        "message": [response.content],
                        "user_suggestions": [user_suggestion],
                    },
                )
            except Exception as e:
                log(f"[medium_purple3]LOG: Failed to parse user suggestion ({user_suggestion}): {e}[/]")
                user_approval, user_suggestion = chat_utility.take_user_feedback(
                    "responseNode",
                    prompt_suffix=(
                        "Invalid input!\n"
                        "Please reply with Yes to approve, or specify a node to re-run:\n"
                        "taskRunnerNode or queryBreakerNode."
                    )
                )
                if user_approval is True:
                    return {
                        "message": [response.content],
                        "finalresponse": response.content,
                    }

    def build_graph(self):
        workflow = StateGraph(ToolAgentState)

        workflow.add_node("queryBreakerNode", self.queryBreakerNode)
        workflow.add_node("taskRunnerNode", self.taskRunnerNode)
        workflow.add_node("responseNode", self.responseNode)

        workflow.add_edge(START, "queryBreakerNode")
        # workflow.add_edge("queryBreakerNode", "taskRunnerNode")
        # workflow.add_edge("taskRunnerNode", "responseNode")
        # workflow.add_edge("responseNode", END)

        return workflow.compile(name=self.agent_name)
    
    
    def graph_invoke(self, query):
        inputs = {
            "message": [
                chat_utility.build_message_structure(role="user", message=query)
            ],
            "query": query,
            "user_suggestions" : [],
            "tool_response_list" : [],
            "finalresponse" : "",
            "task_tool_pairs" : []
        }
        theme_utility.print_startup_info(self.agent_name, self.agent_description, True)
        result = self.graph.invoke(inputs)
        log("[medium_purple3]LOG: Graph execution complete[/]")
        return result

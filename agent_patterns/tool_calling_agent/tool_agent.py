import ast

from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from typing import List, Literal
from pydantic import BaseModel, Field
from langgraph.types import Command

from utils import utility
from utils import chat_utility
from utils import theme_utility
from utils.theme_utility import console, log
from agent_patterns.states import ToolAgentState


ToolPrompts = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "ToolAgent",
)


class TaskToolPair(BaseModel):
    task: str
    suggested_tool: str
    reason: str
    tool_args: dict


class ToolAnalysisResponse(BaseModel):
    user_query: str
    task_tool_pairs: List[TaskToolPair]
    conversational_reply: str


class NodeRoutes(BaseModel):
    node: Literal["queryBreakerNode", "taskRunnerNode", "RerunRemainingToolsNode"]


class ToolAgent:
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        tools: list,
        model: str,
        structure=None,
        log_path = "tool_agent.log",
        context: list = [],
    ):
        log(f"[medium_purple3]LOG: Initializing ToolAgent: {agent_name}[/]")
        self.agent_name = agent_name
        self.agent_description = agent_description

        self.tools = tools
        self.tool_desc, self.tool_names = utility.tools_to_action_prompt(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        self.llm = ChatOllama(model=model)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.graph = self.build_graph()
        self.structure = structure
        self.context = context
        theme_utility.setup_console_logging(log_path)

    # step 1 - break the query into task - tool pair
    async def queryBreakerNode(self, state: ToolAgentState):
        log("[medium_purple3]LOG: Entered queryBreakerNode[/]")
        with console.status(
            f"[plum1] queryBreakerNode setting up...[/]\n", spinner="dots"
        ):
            if len(state["user_suggestions"]) > 0:
                suggestions = "SUGGESTION: " + ", ".join(state["user_suggestions"])
            else:
                suggestions = ""

            prompt = ToolPrompts["queryBreakerPrompt"].format(
                query=state["query"] + suggestions,
                tool_list=utility.tools_to_action_prompt(self.tools),
            )
            structured_resp_llm = self.llm.with_structured_output(ToolAnalysisResponse)
            response = await structured_resp_llm.ainvoke(prompt)

            theme_utility.display_task_list(response.task_tool_pairs)
            theme_utility.display_response(response.conversational_reply)
            log("[medium_purple3]LOG: queryBreakerNode generated response[/]")
            return {
                "task_tool_pairs": response.task_tool_pairs,
                "message": [response.conversational_reply],
            }

    # step 2 - tool runner and storing results
    async def taskRunnerNode(self, state: ToolAgentState):
        log("[medium_purple3]LOG: Entered taskRunnerNode[/]")
        with console.status(f"[plum1] taskRunnerNode setting up...[/]\n", spinner="dots"):
            tool_resp_dict = []
        for task_tool in state["task_tool_pairs"]:
            log(f"[medium_purple3]LOG: Running tool for task={task_tool.task}[/]")
            approved, suggestion = chat_utility.ask_user_approval(
                agent_name="taskRunnerNode",
                prompt_suffix=chat_utility.tool_approval_msg(
                    tool=task_tool.suggested_tool,
                    task=task_tool.task,
                    reason=task_tool.reason,
                    args=task_tool.tool_args,
                ),
            )
            if approved is True:
                result = self.tools_by_name[task_tool.suggested_tool].invoke(
                    task_tool.tool_args
                )
                tool_resp_dict.append(
                    {
                        "task": task_tool.task,
                        "tool": task_tool.suggested_tool,
                        "args": task_tool.tool_args,
                        "result": result,
                    }
                )
            elif approved is False:
                log(f"[medium_purple3]LOG: Tool denied by user[/]")
                tool_resp_dict.append(
                    {
                        "task": task_tool.task,
                        "tool": None,
                        "args": None,
                        "result": "denied",
                    }
                )
            else:                
                log(f"[medium_purple3]LOG: User provided modification suggestion: {suggestion}[/]")
                while True:
                    try:
                        parsed = ast.literal_eval(suggestion)
                        if isinstance(parsed, dict):
                            result = self.tools_by_name[parsed["tool_name"]].ainvoke(
                                parsed["tool_args"]
                            )
                            tool_resp_dict.append(
                                {
                                    "task": task_tool.task,
                                    "tool": parsed["tool_name"],
                                    "args": parsed["tool_args"],
                                    "result": result,
                                }
                            )
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
                            tool_resp_dict.append(
                                {
                                    "task": task_tool.task,
                                    "tool": None,
                                    "args": None,
                                    "result": "denied",
                                }
                            )
                            break
        with console.status(f"[plum1] Generating response...[/]\n", spinner="dots"):
            prompt = ToolPrompts['ToolNodePrompt']
            prompt += f"\n QUERY - {state['query']}"
            prompt += f"\n TOOL RESULTS - {tool_resp_dict}"            
            response = await self.llm.ainvoke(prompt.strip())
        theme_utility.display_response(response.content)
        log(f"[medium_purple3]LOG: Generated tool response[/]")
        user_approval, user_suggestion = chat_utility.ask_user_approval(
            "taskRunnerNode", prompt_suffix = "Do you approve? (yes / no): "
        )
        if user_approval is True:
            return {"message": [response.content], "tool_response_list": tool_resp_dict}
        else:
            log(f"[medium_purple3]LOG: User did not approve final output, rerouting...[/]")
            return Command(
                goto="queryBreakerNode",
                update={"message": [response.content], "user_suggestions": [user_suggestion]},
            )

    # step 2 - tool runner and storing results
    async def RerunRemainingToolsNode(self, state: ToolAgentState):
        log("[medium_purple3]LOG: Entered RerunRemainingToolsNode[/]")
        tool_resp_dict = []
        for tool_result in state["tool_response_list"]:
            if tool_result["tool"] is None:
                prompt = f"Do you want me to add any information about for"
                prompt += f"task - {tool_result['task']}"
                tool_result["result"] = chat_utility.take_user_input(prompt)
                log(f"[medium_purple3]LOG: User provided manual input: {tool_result['result']}[/]")    
            tool_resp_dict.append(tool_result)

        with console.status(f"[plum1] Generating response...[/]\n", spinner="dots"):
            prompt = ToolPrompts['RemainingToolsPrompt']
            prompt += f"\nQUERY : {state['query']}"
            prompt += f"\nTOOL RESULTS : {tool_resp_dict}"
            response = await self.llm.ainvoke(prompt.strip())
        theme_utility.display_response(response.content)
        log("[medium_purple3]LOG: RerunRemainingToolsNode generated response[/]")
        return {"message": [response], "tool_response_list": tool_resp_dict}

    async def finalResponseNode(self, state: ToolAgentState):
        log("[medium_purple3]LOG: Entered finalResponseNode[/]")
        with console.status(f"[plum1] Generating response...[/]\n", spinner="dots"):
            prompt = ToolPrompts["ReportResponsePrompt"]
            prompt += f"\nQUERY : \n{state['query']}"
            prompt += f"\nTOOL RESULTS : \n{state['tool_response_list']}"
            response = await self.llm.ainvoke(prompt.strip())
        theme_utility.display_response(response.content)
        
        log("[medium_purple3]LOG: Final response generated[/]")
        return {
            "message": [response.content],
            "finalresponse": response.content,
        }

    def build_graph(self):
        workflow = StateGraph(ToolAgentState)

        workflow.add_node("queryBreakerNode", self.queryBreakerNode)
        workflow.add_node("taskRunnerNode", self.taskRunnerNode)
        workflow.add_node("RerunRemainingToolsNode", self.RerunRemainingToolsNode)
        workflow.add_node("finalResponseNode", self.finalResponseNode)

        workflow.add_edge(START, "queryBreakerNode")
        workflow.add_edge("queryBreakerNode", "taskRunnerNode")
        workflow.add_edge("taskRunnerNode", "RerunRemainingToolsNode")
        workflow.add_edge("RerunRemainingToolsNode", "finalResponseNode")
        workflow.add_edge("finalResponseNode", END)

        return workflow.compile(name=self.agent_name)

    async def graph_invoke_async(self, query: str):
        inputs = {
            "message": [
                chat_utility.build_message_structure(role="user", message=query)
            ],
            "query": query,
            "task_tool_pairs": {},
            "tool_response_list": [],
            "user_suggestions": []
        }
        theme_utility.print_startup_info(self.agent_name, self.agent_description, True)
        result = await self.graph.ainvoke(inputs)
        log("[medium_purple3]LOG: Graph execution complete[/]")
        return result

from rich.prompt import Prompt
from rich.panel import Panel
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Literal, Optional
from utils.theme_utility import console
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import PydanticOutputParser
import pandas as pd
import re
import json

def build_message_structure(role: str, message :str):
    return {
        'role':role,
        'content':message
        }

def append_to_structure(history:list, role:str, message:str):
    return history.append(build_message_structure(role=role, message=message))

def load_context(context: list[dict[str, str]]):
    combined = []
    for cont in context:
        for name, txt in cont.items():
            combined.append(f"{name} : {txt}")
    combined_txt = "\n".join(combined)
    return f"<CONTEXT>\n{combined_txt}\n</CONTEXT>"


def llm_run(llm_client, messages:list, model:str="llama3.1"):
    if model == "llama3.1":
        response = llm_client.chat(
            message = messages,
            model = model
        )
        return response.message.content

def take_user_feedback(task, prompt_suffix = "Do you approve? (yes / suggest changes): "):
    text = f"[grey39][misty_rose3]{task}[/] awaits your approval[/]"
    console.rule(text, style="grey39")

    while True:
        response = Prompt.ask(f"[grey39]{prompt_suffix}[/]", default="Y").strip()
        if response.lower() in ("yes", "y"):
            console.print(f"[chartreuse2]User approved {task} to proceed.[/]")
            return True, None
        elif response:
            console.print(f"[tan]User suggested changes: {response}.[/]")
            return '', response



def ask_user_approval(agent_name, prompt_suffix="Do you approve? (yes / suggest changes): "):
    text = f"[grey39][misty_rose3]{agent_name}[/] awaits your approval[/]"
    console.rule(text, style="grey39")

    while True:
        response = Prompt.ask(f"[grey39]{prompt_suffix}[/]", default="Y").strip()
        if response.lower() in ("yes", "y"):
            console.print(f"[chartreuse2]User approved {agent_name} to proceed.[/]")
            return True, None
        elif response in ("no", "n"):
            console.print(f"[misty_rose3]User declined {agent_name}.[/]")
            return False, None
        elif response:
            console.print(f"[tan]User suggested changes: {response}.[/]")
            return '', response
        else:
            console.print("[tan]Please type 'yes', 'no', or '/chat'.[/]")

def tool_approval_msg(tool, task, reason, args):
    return f"""
[pale_turquoise1]> Preparing to execute tool:[/] [bold]{tool}[/]
[pale_turquoise1]> Task:[/] {task}
[pale_turquoise1]> Reason:[/] {reason}
[medium_purple2]> Command-line invocation:[/]
[bold]{tool} {args}[/]

[sandy_brown]Confirm action:[/]
  • Type [pale_green1]"Yes"[/] to proceed
  • Type [misty_rose3]"No"[/] to cancel
  • Or provide an updated invocation as a dictionary:
    {{
        tool_name: ..., tool_args: {{ ... }}
    }}
"""

def tool_running_response(tool, args, result):
    console.rule("[grey50]Tool Call", style="grey50")
    console.print(f"[grey35]> Tool:[/] [pale_turquoise1]{tool}[/]")
    console.print(f"[grey35]> Arguments:[/] [light_yellow3]{args}[/]")
    console.print(f"[grey35]> Result:[/] [grey84]{str(result)[:50]}...[/]")

def take_user_input(prompt: str, default: Optional[str] = None):
    if default is not None:
        return Prompt.ask(f"[dark_olive_green1]{prompt}[/]", default=default).strip()
    else:
        return Prompt.ask(f"[dark_olive_green1]{prompt}[/]").strip()

def parse_user_command(user_input: str, pattern:str = r"^/(\w+)(?:\s+(.*))?$"):
    match = re.match(pattern, user_input.strip(), re.IGNORECASE)
    if match:
        command = match.group(1).lower()
        suggestion = match.group(2).strip() if match.group(2) else ""
        return command, suggestion.strip()
    return None, None

def chat_with_bot(history:list,llm, task_tool_dict: str = {}):
    class BotReplies(BaseModel):
        assistant_reply: str = Field(description="The assistant's full reply to the user.")
        decision: Literal['approve', 'reject', 'suggest'] = Field(description="The decision derived from the assistant's reply.")
        action: Optional[str] = Field(default=None, description="The name of the next action/tool to execute and its arguments, if applicable.")
    class ToolBotReplies(BaseModel):
        assistant_reply: str = Field(description="The assistant's full reply to the user.")
        tool: str = Field(description="The name of the tool the assistant recommends using.")
        tool_args: dict = Field(description="Arguments to be passed to the tool.")


    if task_tool_dict:
        tool_name = task_tool_dict["suggested_tool"]
        task = task_tool_dict["task"]
        reason = task_tool_dict["reason"]
        args = task_tool_dict["tool_args"]

        messages = history.copy()
        messages.insert(0, {
            "role": "system",
            "content": (
                f"You are a customer support agent and data analyst specializing in data validation for an e-commerce Market Mix Modelling (MMM) platform. "
                f"You are assisting with deciding whether to use the tool **{tool_name}** "
                f"for the task: *{task}*, because: {reason}.\n"
                f"Here are the current arguments:\n{args}\n"
                f"The user has asked for help via @bot. Work with them to finalize what should be done."
            )
        })
        while True:
            user_message = Prompt.ask("[magenta]You[/]").strip()
            if user_message.lower() in {"skip", "exit"}:
                console.print("[bright_black]Exiting bot chat...[/]")
                break
            messages.append({"role": "user", "content": user_message})
            response = llm.with_structured_output(ToolBotReplies).invoke(messages)
            messages.append({"role": "assistant", "content": response.assistant_reply})

            console.print(f"\n[green]Bot[/]: [bold blue]{response.assistant_reply}[/]")
            console.print(f"[bold blue]Suggested Tool[/]: [bright_cyan]{response.tool}[/]")
            console.print(f"[bold blue]With Arguments[/]: [yellow]{response.tool_args}[/]")

            approval = Prompt.ask("[magenta]Approve this tool usage? (Y/N): [/]", default="Y").strip().lower()
            if approval == "y":
                return {
                    "user_message": user_message,
                    "tool": response.tool,
                    "tool_args": response.tool_args
                }
    else:
        console.print(Panel("Entering free-form assistant chat mode. Type 'exit' or skip to leave.", border_style="cyan"))

        messages = history.copy()
        messages.insert(0, {
            "role": "system",
            "content": (
                "You are a customer support agent and data analyst specializing in data validation for an e-commerce Market Mix Modelling (MMM) platform. "
                "Your role is to help users verify, clean, and ensure the accuracy of their marketing and sales data before analysis. "
                "Using the conversation history below, provide clear, precise, and helpful guidance to assist the user with their data validation questions or issues."
            )
        })
        while True:
            user_message = Prompt.ask("[magenta]You[/]").strip()
            if user_message.lower() in {"skip", "exit"}:
                console.print("[bright_black]Exiting bot chat...[/]")
                break

            messages.append({"role": "user", "content": user_message})
            response = llm.with_structured_output(BotReplies).invoke(messages)
            messages.append({"role": "assistant", "content": response.assistant_reply})

            console.print(f"\n[green]Bot[/]: [bold blue]{response.assistant_reply}[/]")

            if response.decision == "approve":
                console.print(f"[green]Bot (ACTION): [/]: [yellow]{response.action}[/]")
                approval = Prompt.ask("[magenta]Approve this (Y/N): [/]", default="Y").strip()
                if approval.lower() == "y":
                    return {
                        "user_message": user_message,
                        "action": response.action
                    }
                else:
                    console.print("Okay, not approving. You can suggest changes or ask for alternatives.\n", style="bright_black")


def node_approver(node):
    user_message = Prompt.ask(f"[cyan]Routing to[/] [blue]{node}[/], [yellow]approve? (Y/N)[/]").strip().lower()
    if user_message == "y":
        console.print(f"[green]Approved.[/] Proceeding to [blue]{node}[/]...\n")
        return {"next_node": node}
    else:
        options = ["queryBreakerNode", "taskRunnerNode", "RerunRemainingToolsNode"]
        while True:
            selected_node = Prompt.ask(f"[yellow]Next node to go to? Choose one of[/] {options}:\n[green]>[/] ").strip()
            if selected_node in options:
                return {"next_node": selected_node}
                break
            else:
                console.print("[red]Invalid node name. Please type exactly as shown.[/]\n")

import os

def user_input_excel(df_to_edit:pd.DataFrame,file_path:str):
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df_to_edit.to_excel(file_path, index=False)
    console.print(f"\n[green3]Created new Excel file:[/] [bold]{file_path}[/bold]")
    console.print("\n[bold yellow]Opening the Excel file...[/bold yellow] Edit it and then come back here.")
    os.startfile(file_path) 
    Prompt.ask("\n[medium_purple3]Press ENTER when you're done editing the Excel file[/]", default="")
    updated_df = pd.read_excel(file_path)
    console.print("\n[green3]Here's the updated data:[/]\n")
    console.print(updated_df)
    updated_df.to_excel(file_path, index=False)



def parse_json_pydantic(content, pydantic_object, llm):
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        json_part = match.group()
        parser = OutputFixingParser.from_llm(
                        parser=PydanticOutputParser(pydantic_object=pydantic_object),
                        llm=llm
                    )
        structured = parser.parse(json_part)
        return structured
    else:
        return None

def parse_json_from_response(text: str) -> dict | None:
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


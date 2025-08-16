from rich.prompt import Prompt
from typing import Optional
from utils.theme_utility import console
from utils import theme_utility
import pandas as pd
import re
import json
import os
import ast

def build_message_structure(role: str, message :str):
    return {
        'role':role,
        'content':message
        }

def append_to_structure(history:list, role:str, message:str):
    return history.append(build_message_structure(role=role, message=message))

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
            console.print("[tan]Please type 'yes', 'no', or 'Enter suggestion'.[/]")

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
    theme_utility.rich_print_df(updated_df)
    updated_df.to_excel(file_path, index=False)

def parse_json_from_response(text: str) -> dict | None:
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        
        try:
            return ast.literal_eval(match.group(0))
        except Exception:
            return None
    return None


    


# utils/colors.py
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
import os
import socket
from datetime import datetime


from colorama import Fore, Style, init
init(autoreset=True)

def system_message(text: str) -> str:
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

def system_input(text: str) -> str:
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"

# ==================from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich.text import Text
from rich.style import Style
from datetime import datetime
from rich.columns import Columns
import time

console = Console()

# Gemini CLI color scheme
STYLES = {
    "ai": Style(color="grey82", bgcolor="grey11"),
    "assistant": Style(color="light_sky_blue1", bgcolor="grey11", bold=True),
    "user": Style(color="plum1", bgcolor="grey11", bold=True),
}

def print_message(role: str, text: str):
    titles = {
        "ai": "[bold grey70]AI Message[/]",
        "assistant": "[bold light_sky_blue1]Assistant[/]",
        "user": "[bold plum1]User[/]"
    }
    border_colors = {
        "ai": "grey27",
        "assistant": "light_sky_blue1",
        "user": "plum1"
    }
    style = STYLES.get(role, STYLES["ai"])
    panel = Panel(Text(text, style=style), title=titles.get(role, "Message"), border_style=border_colors.get(role, "grey27"))
    console.print(panel)

def print_startup_info(agent_name: str, agent_description: str, is_interactive: bool):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"[bold plum1]Starting {agent_name}[/bold plum1]\n[grey50]{now}[/grey50]"
    console.print(Panel(Align.center(header), style="plum1", width=60))
    table = Table(show_header=False, box=None, pad_edge=False, expand=False)
    table.add_row("[bold medium_purple3]Description[/bold medium_purple3]", agent_description)
    table.add_row("[bold medium_purple3]Interactive[/bold medium_purple3]", str(is_interactive))
    table.add_row("[bold medium_purple3]Startup Time[/bold medium_purple3]", now)
    console.print(table)
    console.print(Panel("[light_sky_blue1]Initializing components...[/light_sky_blue1]", style="light_sky_blue1", width=60))

def ask_user_approval(agent_name, prompt_suffix="to proceed (yes/no/suggest what changes you need): "):
    print_message("user", f"{agent_name} awaits your approval {prompt_suffix}")
    while True:
        response = input("> ").strip().lower()
        if response in ("yes", "y"):
            print_message("assistant", f"User approved {agent_name} to proceed.")
            return True, None
        elif response in ("no", "n"):
            print_message("assistant", f"User declined {agent_name}. Need to retry.")
            return False, None
        elif response:
            print_message("assistant", f"User suggested changes: {response}")
            return '', response
        else:
            print_message("assistant", "Please enter 'yes', 'no', or a suggestion.")

def print_dict_nicely(data: dict,title:str = "Data Snapshot"):
    table = Table(title=title, style="grey82")
    table.add_column("Key", style="light_sky_blue1", no_wrap=True)
    table.add_column("Value", style="grey82")
    for k, v in data.items():
        table.add_row(str(k), str(v))
    console.print(table)

def print_markdown_content(md_text: str):
    md = Markdown(md_text, style="grey82")
    console.print(md)

def print_items_as_panels(items, border_style="grey50", expand=True, column_first=False):
    console = Console()
    panels = [Panel(str(item), expand=True, border_style=border_style) for item in items]
    console.print(Columns(panels, expand=expand, column_first=column_first))

def agent_workflow(agent_name: str):
    print_startup_info(agent_name, f"{agent_name} processes data", is_interactive=True)

    console.log(f"[light_sky_blue1]▸ {agent_name} initialization started[/]")
    with console.status(f"[plum1]{agent_name}: setting up...[/]", spinner="dots"):
        time.sleep(1)

    console.log(f"[plum1] {agent_name} initialized, awaiting approval[/]")

    approved, suggestion = ask_user_approval(agent_name)
    if approved is True:
        console.log(f"[plum1] {agent_name} approved, printing data dictionary[/]")
    elif approved is False:
        console.log(f"[red3] {agent_name} aborted by user.[/]")
        return
    else:
        console.log(f"[medium_purple3] Suggestion received: {suggestion}[/]")
        return

    example_data = {
        "user": agent_name,
        "action": "process_data",
        "status": "pending",
        "items": 5
    }
    print_dict_nicely(example_data)

    console.print(f"\n[light_sky_blue1]Now printing markdown content:[/]\n")
    example_markdown = f"""
# Workflow Update for {agent_name}
The **agent** has processed the initial data.
    
- Step 1: Initialization
- Step 2: User Approval
- Step 3: Data Presentation
- Step 4: Finalization
"""
    print_markdown_content(example_markdown)

    console.print(f"\n[plum1]Final user approval to finish the workflow[/]")
    approved, suggestion = ask_user_approval(agent_name, prompt_suffix="to finish the workflow (yes/no/suggest what changes you need): ")

    if approved is True:
        console.log(f"[plum1] {agent_name} final approval received, working...[/]")
    elif approved is False:
        console.log(f"[red3] {agent_name} finalization aborted by user.[/]")
        return
    else:
        console.log(f"[medium_purple3]✏ Final suggestion received: {suggestion}[/]")
        return

    with Progress(
        SpinnerColumn(style="medium_purple3"),
        TextColumn("[progress.description]{task.description}", style="grey50"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"[plum1]{agent_name} working...[/]", total=5)
        for _ in range(5):
            time.sleep(0.5)
            progress.update(task, advance=1)

    console.log(f"[light_sky_blue1] {agent_name} completed.[/]")

def main():
    console.print(f"\n[bold underline light_sky_blue1]🚀 Multi-Agent Workflow Started[/]\n")
    agents = ["agent1", "agent2", "agent3"]
    for agent in agents:
        agent_workflow(agent)
    console.print(f"\n[plum1]🎉 All agents finished![/]")

if __name__ == "__main__":
    main()

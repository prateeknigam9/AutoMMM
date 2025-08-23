
import asyncio
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.console import Console
from datetime import datetime
from rich.align import Align
from rich.table import Table
from rich.columns import Columns
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich import box
import os
from typing import List, Tuple
import shutil
terminal_width = shutil.get_terminal_size().columns
console_width = int(terminal_width * 0.75)
console = Console(width=console_width)

_file_console = None

def setup_console_logging(log_path: str = "logs/tool_agent.log"):
    global _file_console
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _file_console = open(log_path, "a", encoding="utf-8")

def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"{timestamp} | {message}"
    console.log(full_msg)
    if _file_console:
        _file_console.write(full_msg + "\n")
        _file_console.flush()

def print_logo():

    ascii_art = """\n\n
 █████╗ ██╗   ██╗████████╗ ██████╗ ███╗   ███╗███╗   ███╗███╗   ███╗
██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗████╗ ████║████╗ ████║████╗ ████║
███████║██║   ██║   ██║   ██║   ██║██╔████╔██║██╔████╔██║██╔████╔██║
██╔══██║██║   ██║   ██║   ██║   ██║██║╚██╔╝██║██║╚██╔╝██║██║╚██╔╝██║
██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║██║ ╚═╝ ██║
╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝
"""
    styled_text = Text(ascii_art, style="yellow", justify="center")
    console.print(styled_text+"\n")
    banner = """
# AUTOMMM: AI-Powered Executive Workflow Orchestrator for Market Mix Modeling

**Project Description**  
AUTOMMM is an enterprise-grade AI system that orchestrates complete end-to-end workflows for Market Mix Modeling (MMM) through intelligent multi-agent collaboration. Built with LangGraph and LangChain, it features a hierarchical CEO agent architecture that coordinates specialized teams for data preparation, model execution, and business insights generation.

**Key Features**
- **Executive Orchestration**: CEO agent manages complete workflow lifecycle from data to insights
- **Multi-Team Architecture**: Specialized teams for Data Handling, Model Execution, and Contribution Analysis
- **Intelligent Workflow Management**: Automated phase transitions, progress tracking, and quality assurance
- **Strategic Oversight**: Executive-level reporting, project overviews, and stakeholder summaries
- **Seamless Integration**: Unified interface coordinating LangGraph workflows across all specialized agents
- **Professional-Grade Coordination**: Human-in-the-loop control with full traceability and audit trails

**Use Case**  
Perfect for marketing analytics teams, data scientists, and business stakeholders who need a comprehensive, executive-level solution for MMM workflows. Provides both automated execution and strategic oversight, making it ideal for organizations requiring professional-grade marketing mix modeling with transparent, coordinated workflows and executive-level insights.
"""
    console.print(Panel(Markdown(banner), title="MARKET MIX MODELLING WITH AGENTIC AI", border_style="bright_yellow", title_align="left"))


def show_instructions(commands: List[Tuple[str, str]], title: str = "Agent Control Panel"):
    table = Table(
        show_lines=False,
        box=box.SIMPLE,
        border_style="grey37"
    )
    
    table.add_column("Command", style="bold sky_blue1", no_wrap=True)
    table.add_column("Description", style="light_slate_grey")

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    panel = Panel.fit(
        table,
        title=f"[b yellow]{title}[/b yellow]",
        border_style="grey54",
    )

    console.print(panel)


def print_startup_info(agent_name: str, agent_description: str, is_interactive: bool):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"[bold plum1]Starting {agent_name}[/bold plum1]\n[grey50]{now}[/grey50]"
    console.print(Panel(Align.center(header), style="plum1"))
    table = Table(show_header=False, box=None, pad_edge=False, expand=False)
    table.add_row("[bold light_sky_blue1]Description[/bold light_sky_blue1]", agent_description)
    table.add_row("[bold light_sky_blue1]Interactive[/bold light_sky_blue1]", str(is_interactive))
    table.add_row("[bold light_sky_blue1]Startup Time[/bold light_sky_blue1]", now)
    console.print(table)
    console.print()

def display_task_list(tasks):
    table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
    table.add_column("Task", style="medium_orchid3")
    table.add_column("Tool", style="bright_cyan", no_wrap=True)
    table.add_column("Reason", style="bright_black")
    table.add_column("sequence", style="yellow")
    table.add_column("dependency", style="grey82")
    for task in tasks:
        table.add_row(task.task, task.suggested_tool, task.reason, str(task.sequence_id), str(task.dependency))
    console.print(Panel(table,title = "[magenta]TASKS TO PERFORM[/]"))

def display_response(response: str, title: str = "RESPONSE", title_align: str = "left", border_style: str = "light_goldenrod3"
    ):
    md = Markdown(response)
    console.print(Panel(md, title=title, border_style=border_style, title_align=title_align))
    print("")


def print_dictionary(dict_obj:dict, title:str = "Dict"):
    table = Table(title=title, style="grey82")
    table.add_column("Key", style="light_sky_blue1", no_wrap=True)
    table.add_column("Value", style="grey82")
    for k, v in dict_obj.items():
        table.add_row(str(k), str(v))
    console.print("")   
    console.print(table)   

def print_items_as_panels(items, border_style="grey50", expand=False, column_first=False):
    console = Console()
    panels = [Panel(str(item), expand=True, border_style=border_style) for item in items]
    console.print(Columns(panels, expand=expand, column_first=column_first))

def print_rich_table(rows, headers):
    table = Table(show_header=True, header_style="bold light_pink3", show_lines=True)
    for header in headers:
        table.add_column(header, style="light_yellow3")
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)

def rich_print_df(df):
    table = Table(*df.columns, show_header=True, header_style="bold cyan")
    [table.add_row(*map(str, row)) for row in df.itertuples(index=False)]
    console.print(table)

def print_stderr_as_exception(stderr: str):
    """Pretty-print stderr like console.print_exception"""
    if not stderr:
        return
    tb_text = Text(stderr, style="traceback.text")
    panel = Panel(tb_text, title="[bold red]stderr[/bold red]", border_style="red")
    console.print(panel)

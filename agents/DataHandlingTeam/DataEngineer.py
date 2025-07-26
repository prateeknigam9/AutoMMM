"""
Data Engineer
Role: Responsible for loading, validating, and contextualizing raw MMM input data using Excel files and a UI-based interaction, and preparing it for downstream modeling.
Responsibilities:
    - Guide users to upload Excel files and select valid sheets.
    - Capture business context and metadata from users.
    - Store data and context in memory for downstream use.
    - Extract column-level meaning using LLM prompts.
    - Enable corrections and save final inputs for modeling.
"""

from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from agent_patterns.states import DataEngineerState
from utils import utility
from utils.memory_handler import DataStore
from utils import theme_utility
from utils import chat_utility
import pandas as pd
from utils.theme_utility import console, log
import json
import os

DataValidationPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "DataValidationPrompt",
)


class DataEngineerAgent:
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        model: str,
        log_path: str = "logs/data_engineer.log",
    ):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.model = model
        self.llm = ChatOllama(model=self.model)
        self.log_path = log_path
        theme_utility.setup_console_logging(log_path)
        self.graph = self._build_graph()

    def loadingDataNode(self, state: DataEngineerState):
        log("[medium_purple3]LOG: Starting Data loading[/]")
        with console.status(
            f"[plum1] Data Loading Node setting up...[/]\n", spinner="dots"
        ):
            console.print("")
        file_path = chat_utility.take_user_input("Enter the path to your Excel file:")
        while not os.path.isfile(file_path):
            file_path = chat_utility.take_user_input(
                "Invalid path. Please re-enter Excel file path: "
            ).strip()
        excel_file = pd.ExcelFile(file_path)
        console.print("[sandy_brown]sheet_names:[/] ", excel_file.sheet_names)
        sheet_name = chat_utility.take_user_input("[sandy_brown]Sheet:[/] ").strip()
        while sheet_name not in excel_file.sheet_names:
            sheet_name = chat_utility.take_user_input(
                "Invalid sheet name. Please re-enter: "
            ).strip()

        DataStore.set_df("master_data", pd.read_excel(file_path, sheet_name))
        log(
            f"[medium_purple3] saving master data in memory with key[/] - [turquoise4]master_data[/]"
        )

        data_context = chat_utility.take_user_input(
            "Please describe the context of the data (e.g., what is the data about, business context, etc.): "
        )

        DataStore.set_str("data_context", data_context)
        log(
            f"[medium_purple3] saving data context in memory with key[/] - [turquoise4]data_context[/]"
        )

        response = chat_utility.build_message_structure(
            role="assistant", message="data and context data loaded in the memory"
        )
        log("[dark_green]LOG: Data Loading completed[/]")
        return {"messages": [response]}

    def ColumnContextExtractNode(self, state: DataEngineerState):
        log("[medium_purple3]LOG: Starting Column Context Extraction[/]")
        with console.status(
            f"[plum1] Column Context Extraction Node setting up...[/]\n", spinner="dots"
        ):
            master_data = DataStore.get_df("master_data")
            data_context = DataStore.get_str("data_context")
        prompt = DataValidationPrompt["ColumnContextExtraction"]
        prompt += f"\nContext: {data_context}"
        prompt += "\n\nPlease respond only with a valid JSON dictionary."
        message = chat_utility.build_message_structure(role="system", message=prompt)
        messages = [message] + [
            chat_utility.build_message_structure(
                role="user",
                message=f"\nData sample (first 5 rows): {master_data.head(5).to_dict(orient='records')}",
            )
        ]
        with console.status(
            f"[plum1] Generating response for column context...[/]\n", spinner="dots"
        ):
            response = self.llm.invoke(messages)
        log(f"[medium_purple3]LOG: Generated column context response[/]")
        try:
            theme_utility.print_dictionary(
                json.loads(response.content), title="Column Context"
            )
        except:
            theme_utility.display_response(response.content)

        os.makedirs("memory", exist_ok=True)
        approved, _ = chat_utility.ask_user_approval(agent_name="Column Context")
        if approved is True:
            utility.save_to_memory_file("data_context.txt", str(data_context))
            utility.save_to_memory_file(
                "column_context.txt", str(response.content).strip()
            )
            log(f"[medium_purple3]LOG: Saved column context to memory[/]")
        else:
            console.print(
                "\n[sandy_brown]Opening the text file... Please edit and save, then return here.[/bold yellow]"
            )
            os.startfile(r"memory\column_context.txt")
            chat_utility.take_user_input(
                "\n[bold blue]Press ENTER when you're done editing the text file[/bold blue]"
            ).strip()

        with open(r"memory\column_context.txt", "r", encoding="utf-8") as f:
            updated_text = f.read()
        DataStore.set_str("data_context", updated_text)
        log(f"[medium_purple3]LOG: Saved data context to memory[/]")
        log("[dark_green]LOG: Column Context Extraction completed[/]")
        assistant_message = chat_utility.build_message_structure(
            role="assistant", message="Saved data context to memory"
        )
        return {
            "messages": [assistant_message],
            "completed": True
            }
    

    def _build_graph(self):
        g = StateGraph(DataEngineerState)
        g.add_node("loadingDataNode", self.loadingDataNode)
        g.add_node("ColumnContextExtractNode", self.ColumnContextExtractNode)

        g.add_edge(START, "loadingDataNode")
        g.add_edge("loadingDataNode", "ColumnContextExtractNode")
        g.add_edge("ColumnContextExtractNode", END)
        return g.compile(name=self.agent_name)

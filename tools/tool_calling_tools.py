from langchain_core.tools import BaseTool, tool
# from langchain_experimental.tools import PythonAstREPLTool

from contextlib import redirect_stdout
from io import StringIO
import pandas as pd

class add_tool(BaseTool):
    name :str = "addition"
    description :str = """
    eg: add_tool : 5 + 3
    returns the sum of two numbers
    """
    def _run(self, a:int, b:int):
        return a + b

class multiply_tool(BaseTool):
    name :str = "multiply"
    description :str = """
    eg: multiply_tool : 5 * 3
    returns the multiplication of two numbers
    """
    def _run(self, a:int, b:int):
        return a * b


class calculate_age(BaseTool):
    name :str = "calculating age"
    description :str = """
    eg: calculate_age : karan
    returns the age of people
    """
    def _run(self, query:str):
        ages = {
            "raj": 93,
            "pranay": 23,
            "krishna" :21
        }
        return ages[query.lower()]


# TODO: move df at one place
df = pd.read_excel("data_to_model.xlsx", sheet_name="Sheet1")

@tool
def execute_python_code_on_df(code: str, df_required: bool) -> str:
    """
    Executes Python code and returns add prints to return the output.
    If the query is related to data, provide 'df' in the execution environment,
    else run code normally without 'df'.
    """
    try:
        io_buffer = StringIO()
         
        env = {}
        if df_required:
            env['df'] = df

        with redirect_stdout(io_buffer):
            exec(code, env)
        return io_buffer.getvalue()
    except Exception as e:
        return f"Error: {type(e).__name__} - {str(e)}"

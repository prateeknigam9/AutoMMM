import pandas as pd
from langchain_groq import ChatGroq

from . import config

def process_config(config: dict):
    master_data = pd.read_excel(config['master_data'],sheet_name=config['sheet_name'])
    llm = ChatGroq(
        model_name=config['model'], 
        temperature=0.1
    )
    column_descriptions = config["data_description"]
    return {
        'master_data' : master_data,
        'llm' : llm,
        'column_descriptions' :column_descriptions
    }

if __name__ == "__main__":
    configuration = process_config(config.configuration)
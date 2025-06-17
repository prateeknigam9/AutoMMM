import pandas as pd
from langchain_groq import ChatGroq

import os
from . import config


from dotenv import load_dotenv
load_dotenv()




def process_config(config: dict):
    
    os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
    os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    master_data = pd.read_excel(config['master_data_path'],sheet_name=config['sheet_name'])
    llm = ChatGroq(
        model_name=config['model'], 
        temperature=0.1
    )
    llm_infograph = ChatGroq(
        model_name=config['llm_infograph'], 
        temperature=0.1
    )

    return {
        'master_data' : master_data,
        'llm' : llm,
        'data_description' :config["data_description"],
        'python310_executable' : config["python310_executable"],
        'master_data_path' : config['master_data_path'],
        'sheet_name' : config['sheet_name'],
        'data_profile_path' : config['data_profile_path'],
        'eda_report_path' : config['eda_report_path'],
        'llm_infograph': llm_infograph
    }

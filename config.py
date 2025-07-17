from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

for env_key in ['GROQ_API_KEY', 'LANGSMITH_API_KEY', 'OPENAI_API_KEY']:
    os.environ[env_key] = os.getenv(env_key, '')

os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "data_analyst"

processed_config = {
    'master_data_path': "data_to_model.xlsx",
    'sheet_name': "Sheet1",
    'master_data': pd.DataFrame
} 

if 'master_data_path' in processed_config and 'sheet_name' in processed_config:
    try:
        processed_config['master_data'] = pd.read_excel(
            processed_config['master_data_path'],
            sheet_name=processed_config['sheet_name']
        )
    except Exception as e:
        processed_config['master_data_error'] = str(e)
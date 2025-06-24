import os
import yaml
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Step 4: Set API keys in environment
for env_key in ['GROQ_API_KEY', 'LANGSMITH_API_KEY', 'OPENAI_API_KEY']:
    os.environ[env_key] = os.getenv(env_key, '')


processed_config = dict(config)  

if 'master_data_path' in config and 'sheet_name' in config:
    try:
        processed_config['master_data'] = pd.read_excel(
            config['master_data_path'],
            sheet_name=config['sheet_name']
        )
    except Exception as e:
        processed_config['master_data_error'] = str(e)



# if 'model' in config:
#     processed_config['llm'] = ChatGroq(model_name=config['model'], temperature=0.1)

# if 'llm_infograph' in config:
#     processed_config['llm_infograph'] = ChatGroq(model_name=config['llm_infograph'], temperature=0.1)

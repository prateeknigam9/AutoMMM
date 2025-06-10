# main.py
import os
import sys
from pathlib import Path
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from autommm.config.process_configuration import process_config
from autommm.config import configuration

from autommm.src.eda_agent import eda_bot

config = process_config(configuration)

eda_report_path = config['eda_report_path']

if __name__ == "__main__":

    print("Running agent.py")
    response = eda_bot.graph.invoke({"input" :"generate the report"})

    # Save the output to a Markdown file
    output_path = Path(eda_report_path)
    output_path.write_text(f"# LLM Response\n\n{response['formatted_report'].content}", encoding="utf-8")
    
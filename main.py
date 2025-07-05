# main.py
import os
from pdb import run
import sys
import logging
from ollama import Client
from groq import Groq
from openai import OpenAI
import asyncio
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import processed_config
from utils.data_loading import data_loading
from utils.data_summary import run_data_summary

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "data_loading.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def main():
    setup_logging()

    # llm_client = Client()
    # llm_client = Groq(
    #             api_key=os.environ.get("GROQ_API_KEY"),
    #         )
    llm_client = OpenAI()

    print("\n" + "🧠 " + "═ " * 50 + " 🧠")
    print("\n" + "🤖  AUTOMMM: Market Mix Modelling with Agentic AI\n".upper())

    print("PROJECT SUMMARY".center(110, "-") + "\n")
    print("This project leverages agentic AI to perform Market Mix Modelling (MMM),\n"
        "allowing businesses to accurately analyze and optimize their marketing\n"
        "spend across multiple channels. By integrating advanced AI decision-making\n"
        "capabilities, AUTOMMM improves predictive accuracy and offers actionable\n"
        "insights for maximizing ROI.\n")

    print("🧠 " + "═ " * 50 + " 🧠\n")

    print("🤖 DATA LOADING (AI) 📊 ".center(50, "-") + "\n")
    data_loading(processed_config, llm_client)
    print(" DATA UNDERSTANDING 📊 ".center(50, "-") + "\n")
    run_data_summary(processed_config)
    print("\n🤖 Starting agent calls now".center(50, "-") + "\n")
    # Manager agent - Orhcestration
        # run
            # data analysis agent - analyse the data, data quality checks
                # USER SUGGESTION  - next step back to data analysis
                # USER APPROVAL - next step insight generating agent
                # UPDATES MEMORY
            # Data insights generation agent - generates and saves insights [OUTPUT]
                # ACCESS MEMORY 
                # USER SUGGESTION  - next step back to data analysis (realtime goal setting QA in loop until final)
                # USER APPROVAL - next step model preperation agent
            # Model Preperation generation agent - sets up the parameters and features and configurations
                # ACCESS MEMORY 
                # USER SUGGESTION  - realtime suggestion + memory + prompt  (realtime goal setting QA in loop until final)
                # USER APPROVAL - next step model runner Agent
                # UPDATES MEMORY
            # Model Runner agent - runs a modelling script and performance analysis
                # RUNS SCRIPT
                # USER SUGGESTION  - next step model preperation agent  (realtime goal setting QA in loop until final)
                # USER APPROVAL - next step model preperation agent
                # UPDATES MEMORY
            # Contribution Calculator agent - runs shaply contributions and runs a check with hypothesis from analysis
                # RUNS SCRIPT
                # USER SUGGESTION  - next step model preperation agent  (realtime goal setting QA in loop until final)
                # USER APPROVAL - next step model preperation agent
                # UPDATES MEMORY
            # Business Insight agent - translates the numbers into Business Insights
                # ACCESS MEMORY 
                # RUNS SCRIPT
                # USER SUGGESTION  - next step CHOOSE AGENT (realtime goal setting QA in loop until final)
                # USER APPROVAL - END
            # Quality debate agent - 
                # Task: Critique outputs, improve reasoning
                # Abilities: Cross-agent communication, consensus-building
            # *Note - Quality debate agent is called after every agent recurring and it runs in loop with previous agent, before any human feedback is taken


if __name__ == "__main__":
    main()




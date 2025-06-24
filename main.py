# main.py
import os
import sys
import logging
from ollama import Client
import asyncio
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import processed_config
from utils.data_loading import data_loading
from utils.data_summary import run_data_summary


# --- Shared control state ---
state = {"paused": False}

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

    llm_client = Client()

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

if __name__ == "__main__":
    main()




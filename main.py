# main.py
import os
import sys
from pathlib import Path
import logging
from ollama import Client
import pandas as pd
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import processed_config
from utilites.data_loading import data_loading


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
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


    data_loading(processed_config, llm_client)

if __name__ == "__main__":
    main()

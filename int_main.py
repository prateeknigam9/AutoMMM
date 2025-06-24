# main.py
import os
import sys
import asyncio
import logging
from ollama import Client
import pandas as pd
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import processed_config
from utils.data_loading import data_loading
from utils.data_summary import run_data_summary

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


async def input_handler(session):
    """
    Async input handler accepting commands or normal input.
    """
    while True:
        with patch_stdout():
            user_input = await session.prompt_async(">>> ")

        # Simple commands: start with slash e.g. /pause, /resume, /exit
        if user_input.startswith("/"):
            cmd = user_input[1:].strip().lower()
            if cmd == "pause":
                state["paused"] = True
                print("[SYSTEM] Paused.")
            elif cmd == "resume":
                state["paused"] = False
                print("[SYSTEM] Resumed.")
            elif cmd == "exit":
                print("[SYSTEM] Exiting...")
                os._exit(0)  # hard exit, could do graceful cleanup
            else:
                print("[SYSTEM] Unknown command:", cmd)
        else:
            # You can extend this to handle normal input globally if needed
            print(f"[INPUT] You typed: {user_input}")


async def interactive_pipeline(llm_client):
    print("\n🧠 " + "═ " * 50 + " 🧠")
    print("🤖  AUTOMMM: Market Mix Modelling with Agentic AI\n".upper())
    print("📦 Loading Data Pipeline...\n")

    # Wait if paused
    while state["paused"]:
        await asyncio.sleep(1)

    # Run data_loading with async input support
    # Passing session so data_loading can call async input prompts
    await data_loading(processed_config, llm_client)

    # Pause check before next step
    while state["paused"]:
        await asyncio.sleep(1)

    await asyncio.to_thread(run_data_summary, processed_config)

    print("\n✅ All tasks completed.\n")


async def main():
    setup_logging()
    llm_client = Client()

    session = PromptSession()

    # Run both interactive_pipeline and input_handler concurrently
    await asyncio.gather(
        interactive_pipeline(llm_client),
        input_handler(session),
    )


if __name__ == "__main__":
    asyncio.run(main())

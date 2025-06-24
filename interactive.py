import asyncio
import re

# Shared state
state = {
    "paused": False,
    "parameter": 1
}

async def backend_task():
    step = 0
    last_paused = None  # To track changes in pause state

    while True:
        if state["paused"]:
            if last_paused != True:
                print("[PAUSED] Waiting...")
                last_paused = True
        else:
            if last_paused != False:
                print(f"[WORKING] Resumed with parameter = {state['parameter']}")
                last_paused = False

            # Do the work silently (no printing every step)
            # If you want, you can log progress elsewhere or handle results
            step += 1

        await asyncio.sleep(2)

async def process_user_input():
    while True:
        user_input = await asyncio.to_thread(input, ">> ")

        # Very simple natural language processing
        if "pause" in user_input.lower():
            if not state["paused"]:
                state["paused"] = True
                print("[INFO] Paused.")
            else:
                print("[INFO] Already paused.")
        elif "resume" in user_input.lower():
            if state["paused"]:
                state["paused"] = False
                print("[INFO] Resumed.")
            else:
                print("[INFO] Already running.")
        elif "set" in user_input.lower():
            match = re.search(r"set (\d+)", user_input.lower())
            if match:
                state["parameter"] = int(match.group(1))
                print(f"[INFO] Parameter set to {state['parameter']}.")
            else:
                print("[WARN] Could not extract number.")
        else:
            print("[WARN] Command not recognized. Try: 'pause', 'resume', or 'set X'.")

async def main():
    await asyncio.gather(
        backend_task(),
        process_user_input()
    )

if __name__ == "__main__":
    asyncio.run(main())

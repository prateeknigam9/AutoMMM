# utils/colors.py
from colorama import Fore, Style, init
init(autoreset=True)

def system_message(text: str) -> str:
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

def system_input(text: str) -> str:
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"
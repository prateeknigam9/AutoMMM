# main.py
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import try.py to execute its code
# import autommm.src.try
from autommm.src import prtk

if __name__ == "__main__":
    print("Running try.py...")
    print(prtk.keys())
# profile_report.py

import sys
import os
import subprocess
import pandas as pd

# ---- TEMP FIX FOR MODULE IMPORTS ----
# Adds project root (3 levels up from this file) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from autommm.config.process_configuration import process_config
from autommm.config import configuration


config = process_config(configuration)

python310_executable = config['python310_executable']
data_profile_path = config['data_profile_path']
master_data_path = config['master_data_path']
sheet_name = config['sheet_name']


def install_packages():
    try:
        import ydata_profiling
    except ImportError:
        subprocess.check_call([python310_executable, "-m", "pip", "install", "ydata-profiling"])

def main():
    install_packages()

    code = f"""from ydata_profiling import ProfileReport
import pandas as pd

master_data_path = r"{master_data_path}"
data_profile_path = r"{data_profile_path}"
sheet_name = "{sheet_name}"

df = pd.read_excel(master_data_path, sheet_name=sheet_name)
profile = ProfileReport(df, title="Data Profiling Report")
profile.to_file(data_profile_path)
print(f"Profiling report saved to {{data_profile_path}}")
"""

    try:
        # Run the code using subprocess
        subprocess.check_call([python310_executable, "-c", code])
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}: {e}")
    except FileNotFoundError:
        print(f"Error: Python executable not found at {python310_executable}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

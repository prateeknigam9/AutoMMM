"""
Model Execution Specialist
Role: Executes the regression model and produces outputs for evaluation.
Responsibilities:
    - Run the model on the master dataset with historical marketing and sales data.
    - Track key metrics: R², RMSE, coefficient stability, and contribution accuracy.
    - Log execution details and deliver structured outputs for evaluation or tuning.
"""
import subprocess
from utils.theme_utility import console, log
from utils import theme_utility




class RunnerAgent:
    def __init__(self,agent_name, agent_description,  config_path='config.yaml', runner_path='runner.py', log_path: str = "logs/hbr_runner_agent.log",):
        self.config_path = config_path
        self.runner_path = runner_path
        self.agent_name = f"{agent_name}: Paras"
        self.agent_description = agent_description
        self.log_path = log_path
        theme_utility.setup_console_logging(log_path)

    def run(self):
        log("[medium_purple3]LOG: Executing runner.py...[/]")
        with console.status(f"[plum1] Executing runner.py...[/]", spinner="dots"):
            result = subprocess.run(['python', self.runner_path], capture_output=True, text=True)
        if result.returncode != 0:
            log(f"[medium_purple3]LOG [Runner]ERROR[/]")
            theme_utility.print_stderr_as_exception(result.stderr)
            return False, result.stderr
        log("[green3]LOG: Model Execution Completed[/]")
        return True,''

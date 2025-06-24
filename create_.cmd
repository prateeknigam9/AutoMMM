@REM @echo off

@REM REM Create directories
@REM mkdir agents\src
@REM mkdir agents\prompts
@REM mkdir agents\utilites
@REM mkdir memory
@REM mkdir results
@REM mkdir config

@REM REM Create files in agents\src
@REM type nul > agents\src\data_analysis_agent.py
@REM type nul > agents\src\manager_agent.py
@REM type nul > agents\src\model_preparation_agent.py
@REM type nul > agents\src\model_runner_agent.py
@REM type nul > agents\src\quality_debate_agent.py
@REM type nul > agents\src\__init__.py

@REM REM Create files in agents\prompts
@REM type nul > agents\prompts\data_analysis_agent.yaml
@REM type nul > agents\prompts\manager_agent.yaml
@REM type nul > agents\prompts\model_preparation_agent.yaml
@REM type nul > agents\prompts\model_runner_agent.yaml
@REM type nul > agents\prompts\quality_debate_agent.yaml

@REM REM Create __init__.py in agents
@REM type nul > agents\__init__.py

@REM REM Create files in agents\utilites
@REM type nul > agents\utilites\utiltiy.py
@REM type nul > agents\utilites\tools.py

@REM REM Create files in memory
@REM type nul > memory\memory.txt

@REM REM Create files in results
@REM type nul > results\model_results.txt

@REM REM Create files in config
@REM type nul > config\config.yaml
@REM type nul > config\process_config.py

@REM REM Create root-level files
@REM type nul > .env
@REM type nul > .gitignore
@REM type nul > requirements.txt
@REM type nul > agent_base.py
@REM type nul > main.py

@REM echo File structure created successfully.

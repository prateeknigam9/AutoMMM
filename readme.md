# AutoMMM - Automated Marketing Mix Modeling

AutoMMM is an automated solution for Marketing Mix Modeling (MMM) that leverages advanced AI and machine learning techniques to analyze marketing effectiveness and optimize marketing spend.

## 🚀 Features

- Automated data processing and analysis
- Integration with multiple marketing channels
- Advanced AI-powered insights
- Interactive visualizations
- Support for various data formats


1. Clone the repository:
```bash
git clone [your-repository-url]
cd AutoMMM
```

2. Install the package in development mode:
```bash
pip install -e .
```
## 📦 Project Structure
```bash
AutoMMM/
├── autommm/ # Main package directory
│ ├── config/ # Configuration files
│ ├── src/ # Source code
│ ├── data/ # Data files
│ ├── notebooks/ # Jupyter notebooks
│ └── prompts/ # Prompt templates
├── examples/ # Example usage
├── resources/ # Additional resources
├── setup.py # Package configuration
└── requirements.txt # Project dependencies
```

agents
    src
        data_analysis_agent.py
        manager_agent.py
        model_preparation_agent.py
        model_runner_agent.py
        quality_debate_agent.py
        __init__.py
    prompts
        data_analysis_agent.yaml
        manager_agent.yaml
        model_preparation_agent.yaml
        model_runner_agent.yaml
        quality_debate_agent.yaml
    __init__.py
    utilites
        utiltiy.py
        tools.py
memory
    memory.txt
results
    model_results.txt
config
    config.yaml
    process_config.py
.env
.gitignore
requirements.txt
agent_base.py
main.py
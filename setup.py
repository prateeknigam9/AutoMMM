from setuptools import setup, find_packages

setup(
    name="autommm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-groq",
        "langgraph",
        "jupyter",
        "python-dotenv",
        "matplotlib",
        "scipy",
        "openpyxl",
        "crewai",
        "crewai-tools",
        "langgraph-cli[inmem]",
        "langchain_community",
        "langchain_openai",
        "langchain-experimental"
    ],
    author="Prateek Nigam",
    author_email="p.nigam1@universityofgalway.ie",
    description="AutoMMM - Automated Marketing Mix Modeling",
    python_requires=">=3.13",
)
import yaml


def load_prompt_config(path: str, key: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)[key]
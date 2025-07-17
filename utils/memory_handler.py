import pandas as pd

class DataStore:
    _dfs: dict[str, pd.DataFrame] = {}
    _strings: dict[str, str] = {}

    @classmethod
    def set_df(cls, key: str, df: pd.DataFrame):
        cls._dfs[key] = df
    
    @classmethod
    def set_str(cls, key: str, value: str):
        cls._strings[key] = value

    @classmethod
    def get_df(cls, key: str) -> pd.DataFrame:
        if key not in cls._dfs:
            raise ValueError(f"DataFrame with key '{key}' not found.")
        return cls._dfs[key]
    
    @classmethod
    def get_str(cls, key: str) -> str:
        if key not in cls._strings:
            raise ValueError(f"String value with key '{key}' not found.")
        return cls._strings[key]

    @classmethod
    def clear_all(cls):
        cls._dfs.clear()
        cls._strings.clear()


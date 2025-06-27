class AgentBase:
    def __init__(self, client):
        self.client = client
    
    def trigger(self, prompt : str, system = None):
        raise NotImplementedError
    
    def think(self, *args, **kwargs):
        raise NotImplementedError

    def act(self, *args, **kwargs):
        raise NotImplementedError

    def observe(self, *args, **kwargs):
        raise NotImplementedError

    def communicate(self, *args, **kwargs):
        raise NotImplementedError
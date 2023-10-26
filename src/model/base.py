import yaml
from typing import Dict, Any


class BaseModel:
    ''' The base class of large language models
    
    Attributes:
        require_system_prompt (bool): whether the prompt of the chat model supports system prompt.
    '''
    require_system_prompt: bool

    def load_config(self, config: str|dict) -> Dict:
        ''' Load the model config file, which contains the following attributes:
        
        '''
        if isinstance(config, dict):
            return config
        
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        return config

    def process_fn(self, **kwargs):
        raise NotImplementedError

    def generate(self, data: Any):
        raise NotImplementedError

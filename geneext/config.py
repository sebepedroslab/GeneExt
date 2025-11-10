import yaml 
class Config:
    def __init__(self, config_dict):
        self._load_config(config_dict)

    def _load_config(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    def print_config(self, indent=0):
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                print(' ' * indent + f"{key}:")
                value.print_config(indent + 2)
            else:
                print(' ' * indent + f"{key}: {value}")

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)
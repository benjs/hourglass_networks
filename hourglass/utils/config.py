import yaml
from pathlib import Path


class YAMLConfig:
    """Config class that converts yaml key-value pairs to an pythonic object
    """

    def __init__(self, path: Path, overwrite_dict: dict = {}):
        self._path = path

        with self._path.open('r') as file:
            config_dict = yaml.safe_load(file)

        if "overwrite_config" in config_dict and config_dict['overwrite_config'] != str(path.stem):
            overwrite_path = path.parent / config_dict['overwrite_config']

            if not overwrite_path.exists():
                raise FileNotFoundError(f"Config to overwrite ({overwrite_path}) does not exist!")

            with overwrite_path.open('r') as file:
                self._dict = yaml.safe_load(file)
        else:
            self._dict = {}

        self._dict.update(config_dict)
        self._dict.update(overwrite_dict)

    def __getattr__(self, name):
        if name in self._dict:
            return self._dict[name]

        else:
            raise AttributeError(f"Config does not have parameter {name}.")

    def __str__(self) -> str:
        return yaml.dump(self._dict)

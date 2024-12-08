import os
from dataclasses import dataclass as og_dataclass
from dataclasses import is_dataclass

import yaml


def dataclass(*args, **kwargs):
    """
    Creates a dataclass that can handle nested dataclasses
    and automatically convert dictionaries to dataclasses.
    """

    def wrapper(cls):
        cls = og_dataclass(cls, **kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(value, dict):
                    new_obj = field_type(**value)
                    kwargs[name] = new_obj
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return wrapper(args[0]) if args else wrapper


@dataclass
class LLavaConfig:
    model: str


@dataclass
class DatasetConfig:
    json_path: str
    image_dir: str
    mask_dir: str

    def __post_init__(self):
        self.json_path = os.path.expanduser(self.json_path)
        self.image_dir = os.path.expanduser(self.image_dir)
        self.mask_dir = os.path.expanduser(self.mask_dir)


@dataclass
class SAMConfig:
    model: str
    checkpoint_dir: str
    resize: int
    n_masks: int

    def __post_init__(self):
        self.checkpoint_dir = os.path.expanduser(self.checkpoint_dir)


@dataclass
class AlphaCLIPConfig:
    model: str
    checkpoint_dir: str

    def __post_init__(self):
        self.checkpoint_dir = os.path.expanduser(self.checkpoint_dir)


@dataclass
class ProjectConfig:
    llava: LLavaConfig
    dataset: DatasetConfig
    sam: SAMConfig
    alphaclip: AlphaCLIPConfig


def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))

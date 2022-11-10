import os
import pickle

from omegaconf import OmegaConf

import agents


def load_config(path: str = "config.yaml"):
    config = OmegaConf.load(path)
    return config


def save_agent(_obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(_obj, f)


def load_agent(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def make_agent(name: str):
    if not hasattr(agents, name):
        return None
    return getattr(agents, name)

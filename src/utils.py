import datetime
import functools
import os 

import numpy as np 

SOURCE_DIRECTORY: os.PathLike = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIRECTORY: os.PathLike = os.path.dirname(SOURCE_DIRECTORY)

def get_now_str() -> str:
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

@functools.lru_cache
def get_project_subdirectory(name: str) -> os.PathLike: 
    absolute_path: os.PathLike = os.path.join(PROJECT_DIRECTORY, name)
    if os.path.exists(absolute_path) is False: 
        os.mkdir(absolute_path)
    return absolute_path

LOG_DIRECTORY: os.PathLike = get_project_subdirectory("logs")

@functools.lru_cache
def get_experiment_logs_directory() -> os.PathLike: 
    return get_project_subdirectory("experiment_logs")

def setup_experiment_directory(name: str) -> os.PathLike: 
    now: str = get_now_str()
    prefix_dir: os.PathLike = os.path.join(get_experiment_logs_directory(), name) 

    if os.path.exists(prefix_dir) is False: 
        os.mkdir(prefix_dir)

    directory: os.PathLike = os.path.join(prefix_dir, now)
    assert os.path.exists(directory) is False, f"experiment log directory: {directory} already exists..."
    os.mkdir(directory)

    return directory

normalize: callable = lambda x: x / np.linalg.norm(x)
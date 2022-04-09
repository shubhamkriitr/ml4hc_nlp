import os
from pathlib import Path
from datetime import datetime
from loguru import logger # use this logger (util.logger) in other modules
# Generic utility code

PROJECTPATH = Path(__file__).parent.parent

def get_timestamp_str(granularity=1000):
    if granularity != 1000:
        raise NotImplementedError()
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_")

# Data loader classes for arrythmia and PTB datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from util import PROJECTPATH
import torch
import os

DATASET_LOC_TRAIN = str(Path(PROJECTPATH)/"resources/train.txt")
DATASET_LOC_TEST = str(Path(PROJECTPATH)/"resources/test.txt")
DATASET_LOC_VAL = str(Path(PROJECTPATH)/"resources/dev.txt")
DATASET_DIR_LOC = str(Path(PROJECTPATH)/"resources/")

class TextDataLoaderUtil(object):
    def __init__(self) -> None:
        self.data_loaded = False
    
    def load_raw_text(self, split_name=None, file_path=None):
        """Load based on split_name or path. Loads whole text in memory.
        (no lazy laoding)
        """
        if (split_name is not None) and (file_path is not None):
            raise AssertionError("Both `split_name` and `file_path`"
                                     "cannot be used at the same time.")
        if split_name is not None:
            file_path = self.resolve_path(split_name)
        
        data = None
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        
        return data
    
    def resolve_path(self, split_name):
        valid_names = ["test", "dev", "train"]
        if split_name not in valid_names:
            raise AssertionError(f"`split_name` should be one of these:"
                                 f" {valid_names}")
        return os.path.join(DATASET_DIR_LOC, split_name+".txt")
        
if __name__ == "__main__":
    txt_dataloader = TextDataLoaderUtil()
    raw_txt = txt_dataloader.load_raw_text("test")

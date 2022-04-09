# Data loader classes for arrythmia and PTB datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from util import PROJECTPATH
import torch
import os
from util import logger
from collections import defaultdict

DATASET_LOC_TRAIN = str(Path(PROJECTPATH)/"resources/train.txt")
DATASET_LOC_TEST = str(Path(PROJECTPATH)/"resources/test.txt")
DATASET_LOC_VAL = str(Path(PROJECTPATH)/"resources/dev.txt")
DATASET_DIR_LOC = str(Path(PROJECTPATH)/"resources/")

class TextDataLoaderUtil(object):
    def __init__(self, config={}) -> None:
        self.data_loaded = False
        self.config = config
    
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
    
    def load(self, split_name=None, file_path=None):
        raw_txt = self.load_raw_text(split_name, file_path)
        raw_txt = raw_txt.split("\n")
        
        label_counts = defaultdict(lambda: 0)
        current_record = None
        dataset = []
        
        # for a given record, keep track of labels seen so far
        labels_seen_so_far = set()  

        # last label seen for current record being processed
        last_label_seen = None
        
        for line_num, line in enumerate(raw_txt, 1):
            line = line.strip()
            if line == "" and (current_record is not None):
                dataset.append(current_record)
                current_record = None
                labels_seen_so_far = set()
                last_label_seen = None
                continue
            if line == "":
                assert current_record is None
                continue
            if line.startswith("###"): # new record
                assert current_record is None
                current_record = defaultdict(lambda: [])
                current_record["id"] = int(line.split("###")[1])
                current_record["start_line_num"] = line_num
                continue
            
            # Process label - sentence pair
            assert '\t' in line
            
            label, text_data =  line.split('\t') #assuming there are no tab
            # characters in sentence (text_data)
            
            # do validation related to continuity (i.e. same labels occur in
            # one and only one group)
            if (last_label_seen is not None) and (label in labels_seen_so_far)\
                and (last_label_seen != label):
                logger.error(f"Repeated label group [{label}]"
                             f" at line number {line_num}")
            
            last_label_seen = label
            labels_seen_so_far.add(label)
            
            label_counts[label] += 1
            current_record[label].append(text_data)
        
        logger.info(f"Label counts: {label_counts}")
        return dataset
            
            
            
            
            
                
                
            
            
        
if __name__ == "__main__":
    txt_dataloader = TextDataLoaderUtil()
    raw_txt = txt_dataloader.load_raw_text("test")
    raw_dataset = txt_dataloader.load("test")
    sample = raw_dataset[0]
    print(sample)

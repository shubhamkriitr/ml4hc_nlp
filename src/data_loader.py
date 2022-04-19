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
import Data_loader_nlp
# import datasets

DATASET_LOC_TRAIN = str(Path(PROJECTPATH)/"resources/train.txt")
DATASET_LOC_TEST = str(Path(PROJECTPATH)/"resources/test.txt")
DATASET_LOC_VAL = str(Path(PROJECTPATH)/"resources/dev.txt")
DATASET_DIR_LOC = str(Path(PROJECTPATH)/"resources/")

PUBMED_ID_TO_LABEL_MAP = {0: 'BACKGROUND', 1: 'CONCLUSIONS',
                          2: 'METHODS', 3: 'OBJECTIVE',
                          4: 'RESULTS'}
PUBMED_LABEL_TO_ID_MAP = {label: id_ for id_, label
                          in PUBMED_ID_TO_LABEL_MAP.items()}

class TextDataLoaderUtil(object):
    def __init__(self, config=None) -> None:
        self.data_loaded = False
        if config is None:
            config = {
                "verbose": True,
                "preprocessor_class": None,
                "label_to_id_map": PUBMED_LABEL_TO_ID_MAP,
                "id_to_label_map": PUBMED_ID_TO_LABEL_MAP
            }
        self.config = config
        self.verbose = self.config["verbose"]
        self.label_to_id_map = self.config["label_to_id_map"]
    
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
                current_record = defaultdict(lambda: None)
                current_record["data"] = defaultdict(lambda: [])
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
                and (last_label_seen != label) and self.verbose:
                logger.info(f"Repeated label group [{label}]"
                             f" at line number {line_num}")
            
            last_label_seen = label
            labels_seen_so_far.add(label)
            
            label_counts[label] += 1
            current_record["data"][label].append(text_data)
        
        logger.info(f"Label counts: {label_counts}")
        return dataset
    
    def load_as_text_label_pair(self, split_name=None, file_path=None):
        dataset = self.load(split_name, file_path)
        return self.get_text_label_pairs(dataset)
    
    def get_text_label_pairs(self, dataset):
        text_label_pairs = []
        for idx, record in enumerate(dataset):
            data = record["data"]
            for label in data:
                text_list = data[label]
                text_label_pairs.extend(
                    [(text, self.label_to_id_map[label]) 
                     for text  in text_list]
                )
        
        return text_label_pairs
    
    def get_text(self):
        texts = []
        for split_name in ["train", "dev", "test"]:
            data = self.load_as_text_label_pair(split_name)
            texts.extend([text for text, _ in data])
        return texts

class TransformerDataUtil:
    def __init__(self, datapath):
        self.datapath = Path(datapath)

    def read_data_files(self):
        def read_file(filename):
            labels =[]
            texts =[]
            with open(filename, encoding="utf-8") as f:
                # ordlist = [line.split(None, 1)[0] for line in Training_samples] 
                for line in f:
                    if line!= '\n' and line[0] != '#':
                        label, text = line.split('\t')
                        labels.append(label)
                        texts.append(text)
            return texts, labels
        train_data = read_file(self.datapath / 'train.txt')
        dev_data = read_file(self.datapath / 'dev.txt')
        test_data = read_file(self.datapath / 'test.txt')
        return train_data, dev_data, test_data

    def get_datasets(self):
        training_data, valid_data, test_data = self.read_data_files()
        training_data = datasets.Dataset.from_dict({'text':training_data[0], 'label':list(PUBMED_LABEL_TO_ID_MAP[x] for x in training_data[1])})
        valid_data = datasets.Dataset.from_dict({'text':valid_data[0], 'label':list(PUBMED_LABEL_TO_ID_MAP[x] for x in valid_data[1])})
        test_data = datasets.Dataset.from_dict({'text':test_data[0], 'label':list(PUBMED_LABEL_TO_ID_MAP[x] for x in test_data[1])})
        return training_data, valid_data, test_data
            
            
        
if __name__ == "__main__":
    txt_dataloader = TextDataLoaderUtil()
    raw_txt = txt_dataloader.load_raw_text("test")
    raw_dataset = txt_dataloader.load("test")
    sample = raw_dataset[0]
    txt_label_pairs_1 = txt_dataloader.load_as_text_label_pair("test")
    txt_label_pairs_2 = txt_dataloader.get_text_label_pairs(raw_dataset)
    print(sample)

# Data loader classes for arrythmia and PTB datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from util import PROJECTPATH, BaseFactory
import torch
import os
from util import logger
import util as commonutil
from collections import defaultdict
from constants  import (TOKEN_PAD, TOKEN_PAD_IDX, PUBMED_ID_TO_LABEL_MAP,
                        PUBMED_LABEL_TO_ID_MAP)
# import datasets

DATASET_LOC_TRAIN = str(Path(PROJECTPATH)/"resources/train.txt")
DATASET_LOC_TEST = str(Path(PROJECTPATH)/"resources/test.txt")
DATASET_LOC_VAL = str(Path(PROJECTPATH)/"resources/dev.txt")
DATASET_DIR_LOC = str(Path(PROJECTPATH)/"resources/")
PROCESSED_DATASET_DIR_LOC = str(Path(PROJECTPATH)/"resources/processed_data")


class BaseTextDataLoaderUtil(object):
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
        self.id_to_label_map = self.config["id_to_label_map"]
    
    def load_raw_text(self, split_name=None, file_path=None):
        """Load based on split_name or path. Loads whole text in memory.
        (no lazy laoding)
        """
        raise NotImplementedError()
    
    def resolve_path(self, split_name):
        raise NotImplementedError()
    
    def load(self, split_name=None, file_path=None):
        raise NotImplementedError()
    
    def load_as_text_label_pair(self, split_name=None, file_path=None):
        raise NotImplementedError()
    
    def get_text_label_pairs(self, dataset):
        raise NotImplementedError()
    
    def get_text(self):
        raise NotImplementedError()
class TextDataLoaderUtil(BaseTextDataLoaderUtil):
    def __init__(self, config=None) -> None:
        super().__init__(config=config)
        
    
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

class TextDataLoaderUtilMini(TextDataLoaderUtil):
    """ Loads mini version of dataset files by default,
    otherwise same as `TextDataLoaderUtil`"""
    def __init__(self, config=None) -> None:
        super().__init__(config)
    
    def resolve_path(self, split_name):
        valid_names = ["test", "dev", "train"]
        if split_name not in valid_names:
            raise AssertionError(f"`split_name` should be one of these:"
                                 f" {valid_names}")
        return os.path.join(DATASET_DIR_LOC, split_name+"_mini.txt")


class EmbeddingLoader(object):
    def load(self, embedding_model_path):
        vocab_file_suffix = ".vocab_word_to_index.json"
        logger.info(f"Loading embedding model from {embedding_model_path}")
        
        from gensim.models import Word2Vec
        model = Word2Vec.load(embedding_model_path)
        
        
        logger.info(f"{embedding_model_path}{vocab_file_suffix} file must be present")
        vocab_path = embedding_model_path+vocab_file_suffix
        word_to_index = commonutil.load_json(vocab_path)
        
        self.check_model(model, word_to_index)
        
        # add padding to vocab
        word_to_index[TOKEN_PAD] = TOKEN_PAD_IDX
        assert TOKEN_PAD_IDX == 0 , "Token padding index must be 0"
        
        index_to_word = {idx: word for word, idx in word_to_index.items()}
        embeddings = self.create_embedding_tensor(model, index_to_word)
        
        return embeddings, word_to_index, index_to_word
    
    def create_embedding_tensor(self, model, index_to_word):
        logger.info(f"Total words in vocabulary (including padding) = {len(index_to_word)}")
        num_dims = model.wv[index_to_word[1]].shape[0]
        pad_embedding = np.zeros(shape=(1, num_dims))
        
        embeddings = [pad_embedding]
        # at 0th index - padding was there
        
        for idx in range(1, len(index_to_word)):
            v = model.wv[index_to_word[idx]]
            v = np.expand_dims(v, 0)
            embeddings.append(v)
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        embeddings = torch.tensor(data=embeddings, dtype=torch.float32)
        
        return embeddings
        
    def check_model(self, model, word_to_index):
        logger.debug("Checking model for missing words")
        missing_words = []
        for word in word_to_index:
            try:
                _ = model.wv[word] # check if this word has corresponding embedding
            except KeyError:
                missing_words.append(word)
        
        if len(missing_words) > 0:
            raise AssertionError(f"Embedding model does not have these words:",
                                 f" \n{missing_words}")
        logger.debug("All checks passed.")    
        
class ProcessedTextDataLoaderUtil(BaseTextDataLoaderUtil):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.train_file_name = "processed_train.txt"
        self.val_file_name = "processed_dev.txt"
        self.test_file_name = "processed_test.txt"
        self.label_pipeline = None
        self.text_pipeline = None
        self.device = commonutil.resolve_device()
        self.embedding_model = None
    
    def get_data_loaders(self, root_dir, word_to_index, batch_size, shuffle):
        assert self.text_pipeline is None
        assert self.label_pipeline is None
        
        self.text_pipeline = self.create_text_pipeline(word_to_index)
        # Use PUBMED_LABEL_TO_ID_MAP all throughout
        self.label_pipeline = self.create_label_pipeline(PUBMED_LABEL_TO_ID_MAP)
        
        train_loader = self.create_loader(
            os.path.join(root_dir, self.train_file_name),
            batch_size, shuffle
        )
        val_loader = self.create_loader(
            os.path.join(root_dir, self.val_file_name),
            batch_size, shuffle
        )
        test_loader = self.create_loader(
            os.path.join(root_dir, self.test_file_name),
            batch_size, shuffle
        )
        
        return train_loader, val_loader, test_loader
    
    def create_loader(self, file_path, batch_size, shuffle):
        data = self.load(file_path)
        _loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                   collate_fn=self.collate_batch)
        return _loader
        
    
    def load(self, file_path):
        data = []
        lines = None
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        for l in lines:
            label, text = l.split("\t")
            label = label.strip().upper()
            text = text.strip()
            data.append((label, text))
        
        return data
    def create_text_pipeline(self, word_to_index):
        # it should tokenize and return list of indices
        def _text_pipeline(text):
            text = text.split()
            word_idx = [word_to_index[word] for word in text]
            return word_idx
        
        return _text_pipeline
    
    def create_label_pipeline(self, label_to_index):
        
        def _label_pipeline(self, label):
            return label_to_index[label]

        return _label_pipeline
        
        
    def collate_batch(self, batch):
        # taken from pytorch tutorial
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)
    
# Add newly created specialized loader utils here        
DATALOADER_UTIL_CLASS_MAP = {
    "ProcessedTextDataLoaderUtil": ProcessedTextDataLoaderUtil
}

class DataLoaderUtilFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = DATALOADER_UTIL_CLASS_MAP
    
    def get(self, dataloader_util_class_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
        return super().get(dataloader_util_class_name, config,
            args_to_pass, kwargs_to_pass)

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

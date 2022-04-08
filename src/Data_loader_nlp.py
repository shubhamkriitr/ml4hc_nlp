# Data loader classes for arrythmia and PTB datasets
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from util import PROJECTPATH


Test = "test"
Train = "train"
Validation = "dev"


DATASET_LOC_TRAIN = str(Path(PROJECTPATH)/"resources/train.txt")
DATASET_LOC__TEST = str(Path(PROJECTPATH)/"resources/test.txt")
DATASET_LOC_VALID = str(Path(PROJECTPATH)/"resources/dev.txt")
class DataLoaderUtil:
    def get_data_training(self):
        Training_labels =[]
        Training_data =[]
        with open(DATASET_LOC_TRAIN,'r') as f:
            Training_samples = f.readlines()
            # ordlist = [line.split(None, 1)[0] for line in Training_samples] 
            for i, line in enumerate(Training_samples):
                if line!= '\n':
                    Training_sample_labels = line.split()[0]
                    if Training_sample_labels[0] != '#':
                        Training_labels.append(Training_sample_labels)
                        Training_sample_labels__ = line.split('\t')[1]
                        Training_data.append(Training_sample_labels__)
        return Training_labels,Training_data

    def get_data_test(self):        
        Test_labels=[]
        Test_data=[]
        with open(DATASET_LOC__TEST) as f:
            Test_samples = f.readlines()
            for i, line in enumerate(Test_samples):
                if line!= '\n':
                    Test_sample_labels = line.split()[0]
                    if Test_sample_labels[0] != '#':
                        Test_labels.append(Test_sample_labels)
                        Test_sample_labels__ = line.split('\t')[1]
                        Test_data.append(Test_sample_labels__)
        return Test_labels,Test_data

    def get_data_validation(self):
        Valid_labels=[]
        Valid_data=[]

        with open(DATASET_LOC_VALID) as f:
            Valid_samples = f.readlines()
            for i, line in enumerate(Valid_samples):
                if line!= '\n':
                    Valid_sample_labels = line.split()[0]
                    if Valid_sample_labels[0] != '#':
                        Valid_labels.append(Valid_sample_labels)
                        Valid_sample_labels__ = line.split('\t')[1]
                        Valid_data.append(Valid_sample_labels__)
        return Valid_labels, Valid_data


from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from collections import defaultdict, OrderedDict
from util import logger
from util import PROJECTPATH, get_timestamp_str
from pathlib import Path
from argparse import ArgumentParser
import os
import json


DEFAULT_EMBEDDING_MODEL_PATH \
    = str(Path(PROJECTPATH)/"resources/saved_models/embedding.model")
    
STRINGS_TO_TEST = [
    "nurse",
    "doctor"
    "therapy",
    "treatment",
    "tender",
    "ethnic",
    "male",
    "good",
    "mri",
    "ecg"
]

ANALOGIES = [
    ["man", "boy", "woman"],
    ["woman", "girl", "man"],
    ["man", "doctor", "woman"],
    ["man", "physician", "woman"],
    ["woman", "physician", "man"],
    ["skin", "ointment", "eye"],
    ["eye", "nepafenac", "lung" ],
    ["eye", "myopia", "lung" ]
    
]

def pad_print(msg, width=40, pad_char="="):
    pad_full = max( width - len(msg), 0)
    pad_left = pad_full//2
    pad_right = max(pad_full - pad_left, 0)
    print(pad_char*width)
    print(f"{pad_char*pad_left}{msg}{pad_char*pad_right}")
    print(pad_char*width)

def show_closest_words(model, strings_to_test):
    pad_print("Similar Words")
    for sr, word in enumerate(strings_to_test):
        print(f"{'-'*10}[{sr}]{'-'*10}")
        print(f'>>> "{word}"')
        try:
            similar_words = model.wv.most_similar(word, topn=10)
            print("(similar words, similarity score):")
            for word_sim, score in similar_words:
                print(f"\t {word_sim} , [{score}]")
                
        except KeyError:
            print("This word is not present in model vacabulary")
        print(f"{'-'*22}")
        
def show_analogies(model, analogies_to_test):
    pad_print("Analogies")
    for sr, word_list in enumerate(analogies_to_test):
        print(f"{'-'*10}[{sr}]{'-'*10}")
        a, b, c = word_list
        print(f'>>> {a}->{b}::{c}->?')
        try:
            possible_choices = model.wv.most_similar(
                positive=[c, b], negative=[a,], topn=10)
            print("(possible choices, similarity score):")
            for choice_word, score in possible_choices:
                print(f"\t {choice_word} , [{score}]")
                
        except KeyError:
            print("One of the given words is not present in model vacabulary")
        print(f"{'-'*22}")

def main():
    ap = ArgumentParser()
    ap.add_argument("--embedding-model", "-e", type=str,
                    default=DEFAULT_EMBEDDING_MODEL_PATH)
    
    model_path = ap.parse_args().embedding_model
    
    model = Word2Vec.load(model_path)
    
    show_closest_words(model, STRINGS_TO_TEST)
    show_analogies(model, analogies_to_test=ANALOGIES)


if __name__ == "__main__":
    main()
    
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from collections import defaultdict, OrderedDict
from util import logger
from util import PROJECTPATH
from pathlib import Path
from argparse import ArgumentParser
import os
import json


DEFAULT_CORPUS_FILE_PATH = str(Path(PROJECTPATH)/"resources/processed_data/text_processed_for_learning_embedding.txt")
DEFAULT_EPOCHS = 1000
DEFAULT_OUTPUT_PATH = str(Path(PROJECTPATH)/"resources/saved_models/embedding.model")
DEFAULT_VECTOR_SIZE = 200

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    
class TextCorpusProcessor:
    def read_corpus(self, file_path=DEFAULT_CORPUS_FILE_PATH):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                data.append(line.split())
        return data
    
    def build_vocabulary(self, file_path_or_dataset):
        dataset = None
        if isinstance(file_path_or_dataset, str):
            dataset = self.read_corpus(file_path_or_dataset)
        else:
            dataset = file_path_or_dataset
        vocab = defaultdict(lambda: 0)
        
        for text in dataset:
            for token in text:
                vocab[token] += 1
        
        vocab = OrderedDict(sorted(vocab.items()))
        
        return vocab
    
    @staticmethod
    def create_indexed_vocab(vocab: dict):
        vocab_word_to_index = {}
        vocab_index_to_word = {}
        for idx, word in enumerate(vocab, 1):
            vocab_word_to_index[word]  = idx
            vocab_index_to_word[idx] = word
        
        return vocab_word_to_index, vocab_index_to_word
            
        
        
        

if __name__ == "__main__":
    
    ap = ArgumentParser()
    ap.add_argument("--workers", "-w", type=int, default=8, 
                    help="Number of worker processes to use.")
    ap.add_argument("--corpus-path", "-c", type=str, 
                    default=DEFAULT_CORPUS_FILE_PATH)
    ap.add_argument("--output-path", "-o", type=str, default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--vector-size",
                    "-d", type=int, default=DEFAULT_VECTOR_SIZE)
    
    args = ap.parse_args()
    
    logger.info(f"Provided args: {args}")
    
    corpus_path = args.corpus_path
    epochs = args.epochs
    output_path = args.output_path
    workers = args.workers
    vector_size = args.vector_size
    # set up directories
    
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    txt_corpus_processor = TextCorpusProcessor()
    pubmed_text = txt_corpus_processor.read_corpus(corpus_path)
    vocab = txt_corpus_processor.build_vocabulary(pubmed_text)
    vocab_word_to_index, vocab_index_to_word \
        = txt_corpus_processor.create_indexed_vocab(vocab)
    vocab_sorted_by_frequency = OrderedDict(sorted(vocab.items(), key=lambda x: x[1]))
    model = Word2Vec(sentences=pubmed_text, vector_size=vector_size,
                    window=5, min_count=1, workers=workers,
                    sg=1, # use skip gram
                    hs=1 # use heirarchical softmax
                    )
    model.save(output_path)
    save_json(output_path+".vocab.json", vocab)
    save_json(output_path+".vocab_word_to_index.json", vocab_word_to_index)
    save_json(output_path+".vocab_index_to_word.json", vocab_index_to_word)
    save_json(output_path+".vocab_sorted_by_frequency.json",
              vocab_sorted_by_frequency)
    
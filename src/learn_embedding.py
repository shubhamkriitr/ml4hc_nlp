from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from collections import defaultdict, OrderedDict
from util import logger
from util import PROJECTPATH
from pathlib import Path
from argparse import ArgumentParser
import os

DEFAULT_CORPUS_FILE_PATH = str(Path(PROJECTPATH)/"resources/processed_data/text_processed_for_learning_embedding.txt")
DEFAULT_EPOCHS = 100
DEFAULT_OUTPUT_PATH = str(Path(PROJECTPATH)/"resources/saved_models/word2vec.model")
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
        
        

if __name__ == "__main__":
    
    ap = ArgumentParser()
    ap.add_argument("--workers", "-w", type=int, default=8, 
                    help="Number of worker processes to use.")
    ap.add_argument("--corpus-path", "-c", type=str, 
                    default=DEFAULT_CORPUS_FILE_PATH)
    ap.add_argument("--output-path", "-o", type=str, default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS)
    
    args = ap.parse_args()
    
    logger.info(f"Provided args: {args}")
    
    corpus_path = args.corpus_path
    epochs = args.epochs
    output_path = args.output_path
    workers = args.workers
    # set up directories
    
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    txt_corpus_processor = TextCorpusProcessor()
    pubmed_text = txt_corpus_processor.read_corpus(corpus_path)
    vocab = txt_corpus_processor.build_vocabulary(pubmed_text)
    vocab2 = OrderedDict(sorted(vocab.items(), key=lambda x: x[1]))
    model = Word2Vec(sentences=pubmed_text, vector_size=200,
                    window=5, min_count=1, workers=workers)
    model.save(output_path)
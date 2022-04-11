from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from collections import defaultdict, OrderedDict

CORPUS_FILE_PATH = "/home/shubham/Documents/study/2022SS/ML4HC/projects/local_resources/13f058e9d9dc08d55274482d608988fbea027db7/text_for_training_word2vec.txt"
class TextCorpusProcessor:
    def read_corpus(self, file_path=CORPUS_FILE_PATH):
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
        
        
    
    
txt_corpus_processor = TextCorpusProcessor()
pubmed_text = txt_corpus_processor.read_corpus(CORPUS_FILE_PATH)
vocab = txt_corpus_processor.build_vocabulary(pubmed_text)
vocab2 = OrderedDict(sorted(vocab.items(), key=lambda x: x[1]))
model = Word2Vec(sentences=pubmed_text, vector_size=100,
                 window=5, min_count=1, workers=4)
model.save("word2vec.model")
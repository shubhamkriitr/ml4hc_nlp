from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as nltk_stopwords
import re
import spacy
import string
import tqdm
from constants import(SPACY_MODEL, TOK_NUM, TOKEN_BOS, TOKEN_EOS)

SPECIAL_TOKENS = [
    
]

class BaseTextPreprocessor(object):
    def __init__(self, config=None) -> None:
        if config is None:
            config = {
                "lower_case": True,
                "stem": True,
                
            }   
        self.config = config
        
        
        self.number_regex = re.compile(r'-?[0-9]+[.,]?[0-9]*')
        self.nlp = spacy.load(SPACY_MODEL)
        
        self.init_transformations()
        self.init_stopwords()
        self.init_punctuation_replacement_table()
        # TODO: splitting and joining can be optimized by working on tokens
        
    
    def init_punctuation_replacement_table(self):
        self.punctuations = string.punctuation
        # uncomment this to preserve one `.` instances
        # >>> self.punctuations = string.punctuation.replace(".", "")
        
        # currently removing all punctuations
        self.punctuation_replacement_table \
            = str.maketrans('', '', self.punctuations)
    
    
    def init_stopwords(self):
        self.stopwords = self.nlp.Defaults.stop_words.union(
            nltk_stopwords.words('english'))
          
    def init_transformations(self):
        self.transforms = [
            self.lower_case,
            self.tokenize,
            self.get_replacer(r"%", "percent"),
            self.get_replacer(r"[\.]{1,}", "."), # replace one or more dots 
            # with just one 
            self.lemmatize,
            self.remove_stopwords,
            self.substitute_numbers,
            self.remove_punctuations,
            self.add_bos_eos_tokens,
            self.lower_case, # lower casing again, as some chars were
            # in upper case, during lemmatizing
            self.remove_extra_whitespaces
        ]
        
    def apply_transformations(self, text):
        for transform in self.transforms:
            text = transform(text)
        return text 
    
    def process_dataset(self, text_dataset):
        new_dataset = []
        with tqdm.tqdm(total=len(text_dataset)) as progress_bar:
            for text in text_dataset:
                new_dataset.append(
                    self.apply_transformations(text)
                )
                progress_bar.update(1)
        return new_dataset
    
    def lower_case(self, text: str):
        return text.lower()
    
    def stem(self, text):
        raise NotImplementedError() # implement if needed
    
    def lemmatize(self, text):
        doc = self.nlp(text)
        text = " ".join([token.lemma_.strip() for token in doc
                         if token.lemma_.strip() != ""])
        return text
    
    def substitute_numbers(self, text):
        text = self.number_regex.sub(TOK_NUM, text)
        # making sure that the num tokens are separated (tokenized)
        #tokenize number special tokens
        text = f" {TOK_NUM} ".join(text.split(TOK_NUM))
        
        # remove extra whitespaces
        text = " ".join([word.strip() for word in text.split() if
                            word.strip() != ""])
        return text
    
    def get_replacer(self, regex, new_value):
        prog = re.compile(regex)
        def _replace(text: str):
            text = prog.sub(new_value, text)
            return text
        return _replace
    
    def remove_punctuations(self, text):
        text = [token.translate(self.punctuation_replacement_table)
                    for token in text.split()]
        return " ".join(text)
    
    def remove_stopwords(self, text):
        text = text.strip().split()
        new_text = [token for token in text if token not in self.stopwords]
        return " ".join(new_text)
    
    def remove_extra_whitespaces(self, text):
        text = [token.strip() for token in text.split() if token.strip() != ""]
        return " ".join(text)
    
    def tokenize(self, text):
        """ Tokenizes `text` and combines the tokens back separated by
        just a single whitespace"""
        doc = self.nlp(text)
        tokens = [word.text.strip() for word in doc if word.text.strip() != ""]
        return " ".join(tokens)
    
    def add_bos_eos_tokens(self, text):
        return f"{TOKEN_BOS} {text.strip()} {TOKEN_EOS}"
    

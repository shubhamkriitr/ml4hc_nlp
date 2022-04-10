from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as nltk_stopwords
import re
import spacy
import string

SPACY_MODEL = "en_core_web_lg"
TOK_NUM = "num"
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
        
        
        self.number_regex = re.compile(
            r'([-]?[0-9]+[.,][0-9]*)|([0-9]+[.,][0-9]*)|([-]?[0-9]+)')
        self.nlp = spacy.load(SPACY_MODEL)
        
        self.init_transformations()
        self.init_stopwords()
        self.init_punctuation_replacement_table()
        # TODO: splitting and joining can be optimized by working on tokens
        
    
    def init_punctuation_replacement_table(self):
        # init punctuations to be removed (we want to keep `.`)
        self.punctuations = string.punctuation.replace(".", "")
        self.punctuation_replacement_table \
            = str.maketrans('', '', self.punctuations)
    
    
    def init_stopwords(self):
        self.stopwords = self.nlp.Defaults.stop_words.union(
            nltk_stopwords.words('english'))
          
    def init_transformations(self):
        self.transforms = [
            self.lower_case,
            self.get_replacer(r"%", "percent"),
            self.get_replacer(r"[\.]{1,}", "."), # replace one or more dots 
            # with just one 
            self.lemmatize,
            self.substitute_numbers,
            self.remove_punctuations,
            self.remove_extra_whitespaces
        ]
        
    def apply_transformations(self, text):
        for transform in self.transforms:
            text = transform(text)
        return text 
    
    def process_dataset(self, text_dataset):
        new_dataset = []
        for text in text_dataset:
            new_dataset.append(
                self.apply_transformations(text)
            )
        return new_dataset
    
    def lower_case(self, text: str):
        return text.lower()
    
    def stem(self, text):
        raise NotImplementedError() # implement if needed
    
    def lemmatize(self, text):
        doc = self.nlp(text)
        text = " ".join([token.lemma_ for token in doc])
        return text
    
    def substitute_numbers(self, text):
        text = self.number_regex.sub(TOK_NUM, text)
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
    


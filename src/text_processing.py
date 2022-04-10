from nltk.stem.snowball import SnowballStemmer

class BaseTextPreprocessor(object):
    def __init__(self, config=None) -> None:
        if config is None:
            config = {
                "lower_case": True,
                "stem": True,
                
            }   
        self.config = config
        self.init_transformations()
        
    def init_transformations(self):
        self.transforms = [
            self.lower_case,
            self.get_replacer("%", "percent"),
            
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
    
    def substitute_numbers(self, text):
        # TODO
        return text
    
    def get_replacer(self, value, new_value):
        def _replace(text: str):
            return text.replace(value, new_value)
        return _replace
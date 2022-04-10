from data_loader import TextDataLoaderUtil
from text_processing import BaseTextPreprocessor, TOKEN_BOS, TOKEN_EOS, TOK_NUM
import os
from util import logger
import argparse


class TextCorpusGenerator:
    def __init__(self) -> None:
        """
        sptokNUMPOSVERYLOW
        sptokNUMPOSLOW
        sptokNUMPOS
        sptokNUMPOSHIGH
        sptokNUMPOSVERYHIGH
        sptokNUMNEGVERYLOW
        sptokNUMNEGLOWt
        sptokNUMNEG
        sptokNUMNEGHIGH
        sptokNUMNEGVERYHIGH
        """
        self.test_cases = [
            ("This was a 3-month , multicenter , randomized , open-label study in 335 patients aged 45 years",
             "This was a sptokNUMPOSLOW-month , multicenter , randomized , open-label study in 335 patients aged 45 years")
        ]
    
    
    def run(self, txt_dataloader=None, processor=None, output_dir="."):
        if txt_dataloader is None:
            txt_dataloader = TextDataLoaderUtil()
        if processor is None:
            processor = BaseTextPreprocessor()
        texts = txt_dataloader.get_text() # get text from all splits (`train`,
        # `dev` and `test`)
        os.makedirs(output_dir, exist_ok=True)
        original_text_save_path = os.path.join(
            output_dir, "text_original_lower.txt"
        )
        processed_file_save_path = os.path.join(
            output_dir, "text_processed_for_learning_embedding.txt"
        )
        processed_file_save_path_with_bos_eos = os.path.join(
            output_dir,
            "text_processed_for_learning_embedding_bos_eos_added.txt"
        )
        
        input_ = texts
        self.save_text("\n".join([x.lower() for x in input_]),
                       original_text_save_path)
        result = processor.process_dataset(input_)
        self.save_text("\n".join(result), processed_file_save_path)
        
        self.save_text("\n".join(self.add_bos_eos_tokens_to_corpus(result)),
                       processed_file_save_path_with_bos_eos)
        
        compare = list(zip(input_, result))
        
        return compare
    
    def save_text(self, text, path):
        with open(path, "w") as f:
            f.write(text)
            
    def add_bos_eos_tokens_to_corpus(self, corpus):
        new_corpus = []
        lines = []
        if isinstance(corpus, str):
            with open(corpus, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = corpus
        
        for line in lines:
            new_corpus.append(f"{TOKEN_BOS} {line.strip()} {TOKEN_EOS}")
            
        return new_corpus
    
    def get_corpus_with_number_tokenized(self, corpus_as_list):
        new_corpus = []
        for line in corpus_as_list:
            #tokenize number special tokens
            line = f" {TOK_NUM} ".join(line.split(TOK_NUM))
            
            # remove extra whitespaces
            line = " ".join([word.strip() for word in line.split() if
                             word.strip() != ""])
            new_corpus.append(line)
            
        return new_corpus
            
        
        
        
            

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--output-dir", "-o", type=str, default=".")
    # args = parser.parse_args()
    # TextCorpusGenerator().run(None, None, args.output_dir)
    in_path = "/home/shubham/Documents/study/2022SS/ML4HC/projects/local_resources/txt_corpus_1_6b0c1138674f3c66292c2a5e5fc03a84ab8e0f0c/text_processed_for_learning_embedding.txt"
    out_path = "/home/shubham/Documents/study/2022SS/ML4HC/projects/local_resources/txt_corpus_1_6b0c1138674f3c66292c2a5e5fc03a84ab8e0f0c/text_processed_for_learning_embedding_bos_eos_added.txt"
    r = TextCorpusGenerator().add_bos_eos_tokens_to_corpus(in_path)
    r = TextCorpusGenerator().get_corpus_with_number_tokenized(r)
    TextCorpusGenerator().save_text("\n".join(r), out_path)
    


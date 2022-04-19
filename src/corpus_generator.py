from data_loader import TextDataLoaderUtil
from text_processing import BaseTextPreprocessor, TOKEN_BOS, TOKEN_EOS, TOK_NUM
import os
from util import logger
import argparse


class TextCorpusGenerator:
    def __init__(self) -> None:
        """
        Generates processed text corpus for training embeddings (Word2Vec 
        model etc.)
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

        input_ = texts
        self.save_text("\n".join([x.lower() for x in input_]),
                       original_text_save_path)
        result = processor.process_dataset(input_)
        
        self.save_text("\n".join(result), processed_file_save_path)
        
        
        compare = list(zip(input_, result))
        
        return compare
    
    def save_text(self, text, path):
        with open(path, "w") as f:
            f.write(text)

            
        
        
        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-o", type=str, default=".")
    args = parser.parse_args()
    TextCorpusGenerator().run(None, None, args.output_dir)
    # >>> in_path = "/home/shubham/Documents/study/2022SS/ML4HC/projects/local_resources/txt_corpus_1_6b0c1138674f3c66292c2a5e5fc03a84ab8e0f0c/text_processed_for_learning_embedding.txt"
    # >>> out_path = "/home/shubham/Documents/study/2022SS/ML4HC/projects/local_resources/txt_corpus_1_6b0c1138674f3c66292c2a5e5fc03a84ab8e0f0c/text_processed_for_learning_embedding_bos_eos_added.txt"
    # >>> r = TextCorpusGenerator().add_bos_eos_tokens_to_corpus(in_path)
    # >>> r = TextCorpusGenerator().get_corpus_with_number_tokenized(r)
    # >>> TextCorpusGenerator().save_text("\n".join(r), out_path)

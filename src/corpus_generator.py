from data_loader import TextDataLoaderUtil
from text_processing import BaseTextPreprocessor
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


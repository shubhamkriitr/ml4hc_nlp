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
        original_text_save_path, processed_file_save_path, \
            processed_dev_file_path, processed_train_file_path, \
                processed_test_file_path = self.prepare_output_paths(output_dir)

        input_ = texts
        logger.info(f"Saving all original texts (lower cased) at:"
                    f" {original_text_save_path}")
        logger.info(f"Total samples: {len(input_)}")
        self.save_text("\n".join([x.lower() for x in input_]),
                       original_text_save_path)
        result = []
        
        splits = [
            ("dev", processed_dev_file_path),
            ("train", processed_train_file_path),
            ("test", processed_test_file_path)
        ]
        
        for split_name, output_path in splits:
            text_label_pairs = txt_dataloader.load_as_text_label_pair(
                split_name=split_name, file_path=None
            )# using default loader path
            processed_text = self.process_and_save_with_labels(
                text_label_pairs,
                txt_dataloader.id_to_label_map,
                processor,
                output_path
            )
            result.extend(processed_text)
        
        logger.info(f"Saving processed texts at : {processed_file_save_path}")
        logger.info(f"Total samples: {len(result)}")
        self.save_text("\n".join(result), processed_file_save_path)
        
        compare = list(zip(input_, result))
        
        return compare

    def process_and_save_with_labels(self, text_label_pairs, id_to_label_map,
                                     processor, output_path):
        """Processes text and save text label pairs at `output_path`.
         Returns processed text.
        """
        logger.info(f"Working on : {output_path}")
        logger.info(f"Total samples to process: {len(text_label_pairs)}")
        texts = [ ]
        labels = [ ]
        for t, l in text_label_pairs:
            texts.append(t)
            labels.append(l)
        
        text_label_pairs = []
        # transform texts
        result = processor.process_dataset(texts)
        
        
        for idx in range(len(result)):
            text_label_pairs.append((result[idx], labels[idx]))
        
        
        self.save_text_label_pairs(text_label_pairs, output_path,
                                   id_to_label_map)
        
        return result
        
    def prepare_output_paths(self, output_dir):
        original_text_save_path = os.path.join(
            output_dir, "text_original_lower.txt"
        )
        processed_file_save_path = os.path.join(
            output_dir, "text_processed_for_learning_embedding.txt"
        )
        processed_dev_file_path = os.path.join(
            output_dir, "processed_dev.txt"
        )
        processed_train_file_path = os.path.join(
            output_dir, "processed_train.txt"
        )
        processed_test_file_path = os.path.join(
            output_dir, "processed_test.txt"
        )
        
        return original_text_save_path, processed_file_save_path, \
            processed_dev_file_path, processed_train_file_path, \
                processed_test_file_path
    
    def save_text(self, text, path):
        with open(path, "w") as f:
            f.write(text)
    
    def save_text_label_pairs(self, text_label_pairs, path, id_to_label_map):
        label_and_text_joined = ["\t".join([id_to_label_map[label], text])
                                 for text, label in text_label_pairs]
        self.save_text("\n".join(label_and_text_joined), path)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-o", type=str, default="./text_output")
    args = parser.parse_args()
    TextCorpusGenerator().run(None, None, args.output_dir)
    # >>> in_path = "/home/shubham/Documents/study/2022SS/ML4HC/projects/local_resources/txt_corpus_1_6b0c1138674f3c66292c2a5e5fc03a84ab8e0f0c/text_processed_for_learning_embedding.txt"
    # >>> out_path = "/home/shubham/Documents/study/2022SS/ML4HC/projects/local_resources/txt_corpus_1_6b0c1138674f3c66292c2a5e5fc03a84ab8e0f0c/text_processed_for_learning_embedding_bos_eos_added.txt"
    # >>> r = TextCorpusGenerator().add_bos_eos_tokens_to_corpus(in_path)
    # >>> r = TextCorpusGenerator().get_corpus_with_number_tokenized(r)
    # >>> TextCorpusGenerator().save_text("\n".join(r), out_path)

from data_loader import TextDataLoaderUtil
from text_processing import BaseTextPreprocessor

class TestBaseTextPreprocessor:
    def __init__(self) -> None:
        """
        sptokNUMPOSVERYLOW
        sptokNUMPOSLOW
        sptokNUMPOS
        sptokNUMPOSHIGH
        sptokNUMPOSVERYHIGH
        sptokNUMNEGVERYLOW
        sptokNUMNEGLOW
        sptokNUMNEG
        sptokNUMNEGHIGH
        sptokNUMNEGVERYHIGH
        """
        self.test_cases = [
            ("This was a 3-month , multicenter , randomized , open-label study in 335 patients aged 45 years",
             "This was a sptokNUMPOSLOW-month , multicenter , randomized , open-label study in 335 patients aged 45 years")
        ]
    
    
    def run(self):
        txt_dataloader = TextDataLoaderUtil()
        proc = BaseTextPreprocessor()
        # raw_txt = txt_dataloader.load_raw_text("test")
        # raw_dataset = txt_dataloader.load("test")
        # sample = raw_dataset[0]
        # txt_label_pairs_1 = txt_dataloader.load_as_text_label_pair("test")
        # txt_label_pairs_2 = txt_dataloader.get_text_label_pairs(raw_dataset)
        
        texts = txt_dataloader.get_text()
        
        input_ = texts[10:20]
        result = proc.process_dataset(input_)
        
        compare = list(zip(input_, result))
        
        return compare

if __name__ == "__main__":
    
    TestBaseTextPreprocessor().run()


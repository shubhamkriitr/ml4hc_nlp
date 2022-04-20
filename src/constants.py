SPACY_MODEL = "en_core_web_lg"
TOK_NUM = "spcltokennum"
TOKEN_BOS = "<bos>"
TOKEN_EOS = "<eos>"
TOKEN_PAD = "<pad>" # use index TOKEN_PAD_IDX/0 for it - vocab index should start from 1
TOKEN_PAD_IDX = 0
PUBMED_ID_TO_LABEL_MAP = {0: 'BACKGROUND', 1: 'CONCLUSIONS',
                          2: 'METHODS', 3: 'OBJECTIVE',
                          4: 'RESULTS'}
PUBMED_LABEL_TO_ID_MAP = {label: id_ for id_, label
                          in PUBMED_ID_TO_LABEL_MAP.items()}
# ML for Healthcare Project 2

## Acknowlegment
We expect everything to work on an isolated python environment created 
as per the instructions below, but in case you face any issues running
the code please feel free to contact us by email or on MS-Teams 
(irodrigu@student.ethz.ch, kumarsh@student.ethz.ch, neumannam@ethz.ch).

We have tested our code in an environment with the following specifications:
- Machine:
    - CPU: `11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz`
        - `x86_64 ` 
    - RAM: 16 GB
- OS: `Ubuntu 20.04.4 LTS`
- Python Version: `3.7.11`
## Creating isolated execution environment
- Go to the root directory (after extractig the zip)
- Execute the following in sequence (enter yes when prompted):
```
conda create -n ml4hc_proj2 python=3.7.11
conda activate ml4hc_proj2
pip install -r src/requirements.txt 
```
- Now the environment should be ready
- Make sure to check that the environment is activated before running the code


> **Please make the datasets available in the `resources` folder (see below)
```
(root)
|
├── README.md
├── resources
│   ├── dev_mini.txt
│   ├── dev.txt
│   ├── processed_data
│   │   ├── checksums.txt
│   │   ├── processed_dev.txt
│   │   ├── processed_test.txt
│   │   ├── processed_train.txt
│   │   ├── text_original_lower.txt
│   │   └── text_processed_for_learning_embedding.txt
│   ├── Project2ML4HNLP.pdf
│   ├── saved_models
│   │   ├── checksums.txt
│   │   ├── chectest.txt
│   │   ├── embedding.model
│   │   ├── embedding.model.vocab_index_to_word.json
│   │   ├── embedding.model.vocab.json
│   │   ├── embedding.model.vocab_sorted_by_frequency.json
│   │   └── embedding.model.vocab_word_to_index.json
│   ├── test_mini.txt
│   ├── test.txt
│   ├── train_mini.txt
│   └── train.txt
└── src
    ├── constants.py
    ├── corpus_generator.py
    ├── cost_functions.py
    ├── Data_loader_nlp.py
    ├── data_loader.py
    ├── dev_notebook_ml4hc_nlp.ipynb
    ├── experiment_configs
    │   ├── exp_02b_task2_ann.yaml
    │   ├── exp_02_task2_ann.yaml
    │   ├── exp_03_task2_ann_unfrozen_embeddings.yaml
    │   └── exp_04_task2_cnn_res.yaml
    ├── learn_embedding.py
    ├── main.py
    ├── model_ann.py
    ├── model_baseline.py
    ├── model_cnn_res.py
    ├── model_factory.py
    ├── model_transformer.py
    ├── requirements.txt
    ├── test_embeddings.py
    ├── text_processing.py
    ├── trainingutil.py
    ├── transformer_pipeline.py
    ├── util.py
    └── Word2Vec.ipynb


```
conda create -n ml4hc_proj2 python=3.7.11
conda activate ml4hc_proj2
pip install -r src/requirements.txt 
python src/trainingutil.py --config src/experiment_configs/exp_02_task2_ann.yaml 

```
## Task 2

- To create the preprocessed files riun the folowing command

```
python corpus_generator.py -o corpus_output_folder
```
This will create following files:
```
corpus_output_folder
|
├── processed_dev.txt
├── processed_test.txt
├── processed_train.txt
├── text_original_lower.txt
└── text_processed_for_learning_embedding.txt
```

Files with `processed_` prefix , have label and processed text pairs. 
_e.g._
```
BACKGROUND	<bos> this study analyze liver function abnormality in heart failure patient admit with severe acute decompensate heart failure adhf <eos>
RESULTS	<bos> a post hoc analysis be conduct with the use of datum from the evaluation study of congestive heart failure and pulmonary artery catheterization effectiveness escape <eos>
RESULTS	<bos> liver function test lft be measure at spcltokennum time point from baseline at discharge and up to spcltokennum month follow up <eos>
```

```
ANALOGY:
Bromfenac is a nonsteroidal anti-inflammatory drug marketed in the US as an ophthalmic solution by ISTA Pharmaceuticals for short-term, local use
Nepafenac, sold under the brand name Nevanac among others, is a nonsteroidal anti-inflammatory drug, usually sold as a prescription eye drop 0.1
```


```
defaultdict(<function <lambda> at 0x7fce7b3523b0>, {'OBJECTIVE': 13839, 'METHODS': 59353, 'RESULTS': 57953, 'CONCLUSIONS': 27168, 'BACKGROUND': 21727})
[cn[PUBMED_ID_TO_LABEL_MAP[i]] for i in PUBMED_ID_TO_LABEL_MAP]
[21727, 27168, 59353, 13839, 57953]
np.array(_)
Traceback (most recent call last):
  File "<string>", line 1, in <module>
NameError: name '_' is not defined
np.array([cn[PUBMED_ID_TO_LABEL_MAP[i]] for i in PUBMED_ID_TO_LABEL_MAP])
array([21727, 27168, 59353, 13839, 57953])
x__ = np.array([cn[PUBMED_ID_TO_LABEL_MAP[i]] for i in PUBMED_ID_TO_LABEL_MAP])
x__
array([21727, 27168, 59353, 13839, 57953])
1/(x__)
array([4.60256823e-05, 3.68080094e-05, 1.68483480e-05, 7.22595563e-05,
       1.72553621e-05])
```
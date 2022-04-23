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


# Download spacy model used for text processing (see `text_processing.py`)
python -m spacy download en_core_web_lg
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


# Task 1 (@Amira #TODO)


# Task 2 (@Shubham #TODO)


## Creating processed corpus (Similar to files in: resources/processed_data/)

First make sure that `dev.txt`, `train.txt`, and `test.txt` are available in 
the `resources` directory, and spacy's `en_core_web_lg` model is downloaded.
(`src/text_processing.py` will try to download this automatically when running
 `src/corpus_generator.py`)

  
To create processed corpus for training the embeddings and learning classification
model (for task 2), run the following:
```
python src/corpus_generator.py -o <output directory path>
```
_e.g._ `python src/corpus_generator.py -o new_corpus`

This will create following files:
```
<output directory path>
|
├── processed_dev.txt
├── processed_test.txt
├── processed_train.txt
├── text_original_lower.txt
└── text_processed_for_learning_embedding.txt
```

Files with `processed_` prefix , have label and processed text pairs, while others will have just processed texts.

## Training and testing Word2Vec Model

- **Training the embedding model**

  - The file `resources/processed_data/text_processed_for_learning_embedding.txt`  created in the previous step is used for training the embedding model (by default)

  - To create trained embedding model (Word2Vec)
    ```
      python src/learn_embedding.py
    ```
    This will train Word2Vec model (with vector size 200, and other default parameters)
  - If you want to change the input corpus, output path, vector size, epochs _etc._, then pass them as arguments.
  - Run `python src/learn_embedding.py -h` for argument information.


- **Testing the embedding model**
  - Run:
    ```
    python src/test_embeddings.py
    ```
  - It will load the embedding model from `resources/saved_models/embedding.model`, and using this model it will print out a list of similar words for a few test words like `ecg`, `doctor` _etc._, and after that it will print out analogy results for some word triplets _e.g._ `woman->girl::man->?`


## Training classifier

- To start training execute:
  - ```
    python src/trainingutil.py --config <path-to-run-config-file>
    ```
  - _e.g._
    ```
    python src/trainingutil.py --config src/experiment_configs/eval_02_task2_ann.yaml
    ```
  - The **src/experiment_configs** directory contains many configs, that we have used
  for running our experiments. You can choose any of those or create your own.

The steps above will do the following:
- It will start training 
- create `runs` folder if not already present
- create a timestamped folder with `tag` value provided in the config as suffix
_e.g._ : `2022-04-23_154910__exp_02_task2_ann`
  - this folder will be used to keep track of the model checkpoints, best model etc.
  - in this folder `logs` subfolder will be created in which tensorboard logs will be saved.
- the best model will be saved  if the validation F1 (weighted) has increased when
  compared to the last best F1. Test F1 is also printed in the logs , but validation F1 is
  used for selecting the best model.
### These are the training Config File used for different experiments (training)
|**Config File**| **Experiment description**|
|--------------------|---------------------|
|`exp_02_task2_ann.yaml`|Fully conncted neural network (with class weighting used)|
|`exp_02b_task2_ann.yaml`|Fully connected neural network|
|`exp_03_task2_ann_unfrozen_embeddings.yaml`|Fully connected neural network (with embedding being also fine tuned)|
|`exp_04_task2_cnn_res.yaml`|Residual CNN based model|

## Evaluation of saved classifier

To evaluate the models the script `src/evalutil.py` and the configs in `src/experiment_configs/eval` can be used.

_e.g._


```
python src/evalutil.py --config src/experiment_configs/eval/eval_02_task2_ann.yaml TODO UPDATE saved model here
```

This should print the scores on valdiation and test datasets.

- Make sure that the correct checkpoint path is set in the config file (under the field : `checkpoint_path`) 



# Task 3 (@Ivan #TODO)





----

# Snippets ( TODO: remove later)
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

```
python src/trainingutil.py --config src/experiment_configs/eval_02b_task2_ann.yaml
```
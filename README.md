# NLP

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
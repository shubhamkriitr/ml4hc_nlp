# NLP

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
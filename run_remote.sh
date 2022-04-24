# Activate Conda
eval "$(/cluster/home/kumarsh/miniconda3/bin/conda shell.bash hook)"
conda activate ml4hc_proj2 
cd /cluster/scratch/kumarsh/ml4hc_nlp
python src/corpus_generator.py -o resources/processed_data
python src/learn_embedding.py -o resources/saved_models
python src/trainingutil.py --config src/experiment_configs/exp_02_task2_ann.yaml
python src/trainingutil.py --config src/experiment_configs/exp_02b_task2_ann.yaml
python src/trainingutil.py --config src/experiment_configs/exp_04_task2_cnn_res.yaml
python src/trainingutil.py --config src/experiment_configs/exp_03_task2_ann_unfrozen_embeddings.yaml

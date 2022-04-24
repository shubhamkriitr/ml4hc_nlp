# Activate Conda
eval "$(/cluster/home/kumarsh/miniconda3/bin/conda shell.bash hook)"
conda activate ml4hc_proj2 
cd /cluster/scratch/kumarsh/ml4hc_nlp
python src/trainingutil.py --config src/experiment_configs/exp_04_task2_cnn_res.yaml

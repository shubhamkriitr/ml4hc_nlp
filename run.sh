echo "Running this will overwrite resources/processed_data and resources/saved_models"
read -p "Press 'y' and 'enter' to continue:  " VAR1

VAR2="y"
if [ "$VAR1" = "$VAR2" ]; then
    echo "Starting..."
else
    echo "Exitting..."
    exit 0
fi

python src/corpus_generator.py -o resources/processed_data
python src/learn_embedding.py -o resources/saved_models
python src/trainingutil.py --config src/experiment_configs/exp_02_task2_ann.yaml
python src/trainingutil.py --config src/experiment_configs/exp_02b_task2_ann.yaml
python src/trainingutil.py --config src/experiment_configs/exp_03_task2_ann_unfrozen_embeddings.yaml
python src/trainingutil.py --config src/experiment_configs/exp_04_task2_cnn_res.yaml

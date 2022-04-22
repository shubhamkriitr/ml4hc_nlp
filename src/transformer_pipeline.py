from data_loader import TransformerDataUtil, PUBMED_ID_TO_LABEL_MAP
from datasets import Dataset, DatasetDict, load_metric
from transformers import ProgressCallback, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer_callback import TrainerCallback
import transformers
from util import get_timestamp_str, PROJECTPATH
from torch.utils.tensorboard import SummaryWriter
from transformers.data.data_collator import DefaultDataCollator, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import logging
import os
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
import datasets


logger = logging.getLogger(name=__file__)

class TransformerFactory:
    def get(name):
        return AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(PUBMED_ID_TO_LABEL_MAP))

class TokenizerFactory:
    def get(name):
        return AutoTokenizer.from_pretrained(name)

"""metric_dict = {
    'accuracy': (load_metric('accuracy'), {}),
    'f1_macro': (load_metric('f1'), dict(average="macro"))}"""

TRANSFORMER_DATA = str(Path(PROJECTPATH)/"pubmed-rct-master"/"PubMed_200k_RCT")

class SummarizerCallback(TrainerCallback):
    def __init__(self, summarizer: SummaryWriter):
        self.summarizer = summarizer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.summarizer.add_scalar('train/loss', logs['loss'], logs['epoch'])
            self.summarizer.add_scalar('train/lr', logs['learning_rate'], logs['epoch'])
        elif 'eval_loss' in logs:
            self.summarizer.add_scalar('eval/loss', logs['eval_loss'], logs['epoch'])
            self.summarizer.add_scalar('eval/accuracy', logs['eval_accuracy'], logs['epoch'])
            self.summarizer.add_scalar('eval/f1', logs['eval_f1'], logs['epoch'])
        else:
            logger.info('train_runtime={}'.format(logs['train_runtime']))


class TransformerPipeline:
    def __init__(self, config, device=None) -> None:
        self.config = config
        self.device = "cuda" if device is None else device
        self.prepare_summary_writer()
        self.prepare_datasets()
        self.prepare_model()
        self.prepare_metrics()
        self.prepare_batch_callbacks()
        self.prepare_epoch_callbacks()
        self.prepare_trainer()
        self.save_config() 

    def prepare_datasets(self):
        logger.info("Loading tokenizer {}".format(self.config["tokenizer"]))
        self.tokenizer = TokenizerFactory.get(self.config["tokenizer"])
        if self.config["load_disk_data"]:
            logger.info("Loading data from disk")
            data = datasets.load_from_disk(self.config["load_disk_data"])
            self.training_data = data['train']
            self.valid_data = data['valid']
            self.test_data = data['test']
            logger.info('Samples: train={}, valid={}, test={}'.format(len(self.training_data), len(self.valid_data), len(self.test_data)))
        else:
            logger.info("Loading data from pubmed")
            dataloader = TransformerDataUtil(TRANSFORMER_DATA)
            training_data, valid_data, test_data = dataloader.get_datasets()
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], max_length=self.config["max_length"])
            self.training_data = training_data.map(tokenize_function).remove_columns('text')
            self.valid_data = valid_data.map(tokenize_function).remove_columns('text')
            self.test_data = test_data.map(tokenize_function).remove_columns('text')

        if self.config["save_disk_data"]:
            logger.info("Saving data to disk")
            DatasetDict({
                'train': self.training_data,
                'valid': self.valid_data,
                'test': self.test_data
            }).save_to_disk(Path(self.current_experiment_directory)/'data')

        if self.config["downsample_train"]:
            logger.info("Downsampling training data")
            self.training_data = self.training_data.shuffle(seed=0).select(range(self.config["downsample_train"]))
        if self.config["downsample_valid"]:
            logger.info("Downsampling validation data")
            self.valid_data = self.valid_data.shuffle(seed=0).select(range(self.config["downsample_valid"]))

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )


    def prepare_model(self):

        if self.config["load_from_checkpoint"]:
            raise NotImplementedError()
            checkpoint_path = self.config["checkpoint_path"]
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            #self.model.load_state_dict(torch.load(checkpoint_path))
            logger.info(str(self.model))
            logger.info(f"Model Loaded")
        else:
            logger.info(f"Creating new model")
            model = TransformerFactory.get(self.config["from_pretrained"])
            self.model = model

        modules = list(model.named_modules())
        
        if self.config["fit_modules_from"]:
            logger.info("Fitting modules from {}".format(self.config["fit_modules_from"]))
            unfreeze_from = self.config["fit_modules_from"]
        elif self.config["fit_modules"]:
            raise NotImplementedError()
        else:
            logger.info("Fitting all modules")
            unfreeze_from = 0

        unfrozen = []
        # unfreeze necessary modules
        for _, module in modules[:unfreeze_from]:
            for param in module.parameters():
                param.requires_grad = False
        for name, module in modules[unfreeze_from:]:
            for param in module.parameters():
                param.requires_grad = True
            unfrozen.append(name)

        logger.info("Unfrozen layers {}".format(', '.join(unfrozen)))
        logger.info("Model to {}".format(self.device))
        if self.device == "cpu":
            self.model.cpu()
        else:
            self.model.cuda()

    def prepare_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.current_experiment_log_directory,
            evaluation_strategy=self.config["evaluation_strategy"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            learning_rate=self.config["learning_rate"],
            lr_scheduler_type=self.config["lr_scheduler_type"],
            warmup_ratio=self.config["warmup_ratio"],
            num_train_epochs=self.config["num_epochs"],
            logging_steps=self.config["log_steps"],
            eval_steps=self.config["eval_steps"],
            save_steps=self.config["save_steps"],
            save_strategy=self.config["save_strategy"],
            save_total_limit=self.config["save_total_limit"]
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.training_data,
            data_collator=self.data_collator,
            eval_dataset=self.valid_data,
            compute_metrics=self.compute_metrics
            #callbacks=[SummarizerCallback(self.summary_writer)]
        )

        if self.config["verbose"] == 0:
            self.trainer.remove_callback(ProgressCallback)

    def prepare_metrics(self):
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            res = { }
            res['accuracy'] = accuracy_score(labels, predictions)
            res['f1'] = f1_score(labels, predictions, average='macro')
            #for metric, kwargs in metrics:
            #    metric_val = metric.compute(predictions=predictions, references=labels, **kwargs)
            #    res.update(metric_val)
            return res
            
        self.compute_metrics = compute_metrics

    def filter_trainer_parameters(self):
        trainable_params = []
        trainable_param_names = []
        frozen_params = []
        frozen_param_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                trainable_param_names.append(name)
            else:
                frozen_params.append(param)
                frozen_param_names.append(name)
        
        return trainable_params, trainable_param_names, frozen_params, \
                frozen_param_names

    def prepare_summary_writer(self):
        experiment_tag = self.config["experiment_metadata"]["tag"]
        timestamp = get_timestamp_str()
        self.current_experiment_directory = os.path.join(
            self.config["logdir"],timestamp+"_"+experiment_tag)

        os.makedirs(self.current_experiment_directory, exist_ok=True)
        self.current_experiment_log_directory = os.path.join(
            self.current_experiment_directory, "logs"
        )
        os.makedirs(self.current_experiment_log_directory, exist_ok=True)
        
        #self.summary_writer = SummaryWriter(
        #    log_dir=self.current_experiment_log_directory)
    
    def prepare_batch_callbacks(self):
        self.batch_callbacks = [self.batch_callback]

    def prepare_epoch_callbacks(self):
        self.epoch_callbacks = [self.epoch_callback]

    def run_experiment(self):
        self.trainer.train()
        if self.config["save_pretrained"]:
            logger.info("Saving model")
            self.model.save_pretrained(self.current_experiment_directory+"/final_model")
    
    def batch_callback(self, model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        
        if global_batch_number % self.config["batch_log_frequency"] == 0:
            print(
            f"[({global_batch_number}){current_epoch}-{current_epoch_batch_number}]"
            f" Loss: {kwargs['loss']}")
        #if global_batch_number % self.config["tensorboard_log_frequency"] == 0:
        #    self.summary_writer.add_scalar("train/loss", kwargs['loss'], global_batch_number)
    
    def epoch_callback(self, model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        if current_epoch == 1: # the epoch just finished
            # save the config
            self.save_config()
    
        model.eval()

    def save_config(self):
        try:
            file_path = os.path.join(self.current_experiment_directory,
                                    "config.yaml")
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f)
        except Exception as exc:
            print(exc) # TODO: replace all prints with logger         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("-v", "--verbose", action="store", default=1, type=int)
    args = parser.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    #logger.basicConfig()
    logging.basicConfig(level=logging.ERROR)
    logger.setLevel(logging.INFO)
    with open(args.config, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    config_data["verbose"] = args.verbose
    logger.info('Config:'+str(config_data))
    experiment = TransformerPipeline(config_data, device=device)
    experiment.run_experiment()
    #print(experiment.trainer.log_history)
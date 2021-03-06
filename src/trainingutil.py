import os
import logging
from argparse import ArgumentParser
import torch
import yaml
from torchmetrics import F1Score
from torch import nn
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import (DataLoaderUtilFactory, EmbeddingLoader)
from model_factory import ModelFactory
from util import get_timestamp_str, BaseFactory
import util as commonutil
from util import logger
from cost_functions import CostFunctionFactory
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

F1_AVERAGING_SCHEME = "weighted" # e.g. "micro"/ "macro"/ "weighted"
SEED_VALUE = 100
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
torch.cuda.manual_seed(SEED_VALUE)
torch.backends.cudnn.deterministic = True



class BaseTrainer(object):

    def __init__(self, model: nn.Module, dataloader, cost_function,
                 optimizer: Optimizer,
                batch_callbacks=[], epoch_callbacks=[], config={}) -> None:
        self.model = model
        self.cost_function = cost_function
        self.dataloader = dataloader
        self.batch_callbacks = batch_callbacks
        self.epoch_callbacks = epoch_callbacks

        # read from config: TODO
        self.num_epochs = 100
        self.config = config
        self.num_epochs = self.config["num_epochs"]
        self.optimizer = optimizer


    def train(self):
        global_batch_number = 0
        current_epoch_batch_number = 0
        for current_epoch in range(1, self.num_epochs + 1):
            current_epoch_batch_number = 0
            for batch_data in self.dataloader:
                global_batch_number += 1
                current_epoch_batch_number += 1

                # perform one training step
                self.training_step(batch_data, global_batch_number,
                                    current_epoch, current_epoch_batch_number)
            self.invoke_epoch_callbacks(self.model, batch_data, global_batch_number,
                                current_epoch, current_epoch_batch_number)
            
    def training_step(self, batch_data,  global_batch_number, current_epoch,
                    current_epoch_batch_number):
        
        # make one training step
        
        raise NotImplementedError()

    def invoke_epoch_callbacks(self, model, batch_data, global_batch_number,
                                current_epoch, current_epoch_batch_number):
        self.invoke_callbacks(self.epoch_callbacks, 
                    [self.model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number], {})

    def invoke_callbacks(self, callbacks, args: list, kwargs: dict):
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                logger.exception(exc)

class NetworkTrainer(BaseTrainer):

    def training_step(self, batch_data, global_batch_number,
                        current_epoch, current_epoch_batch_number):
        # make sure training mode is on 
        self.model.train()

        # reset optimizer
        self.optimizer.zero_grad()

        # unpack batch data
        x, y_true = batch_data

        # compute model prediction
        y_pred = self.model(x)

        # compute loss
        loss = self.cost_function(input=y_pred, target=y_true)

        # backward pass
        loss.backward()

        # take optimizer step
        self.optimizer.step()

        self.invoke_callbacks(self.batch_callbacks, 
                    [self.model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number], {"loss": loss})


class BaseExperimentPipeline(object):
    """Class to link experiment stages like
    training, logging, evaluation, summarization etc.
    """

    def __init__(self, config) -> None:
        self.config = None
        self.initialize_config(config)
    
    def initialize_config(self, config):
        config = self.load_config(config)

        # TODO: add/ override some params here
        self.config = config


    def prepare_experiment(self):
        raise NotImplementedError()

    def run_experiment(self):
        raise NotImplementedError()

    def load_config(self, config):
        if isinstance(config, dict):
            return config
        if isinstance(config, str):
            config_data = {}
            with open(config, "r", encoding="utf-8") as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            return config_data

# dictionary to refer to class by name
# (to be used in config)
TRAINER_NAME_TO_CLASS_MAP = {
    "NetworkTrainer": NetworkTrainer
}

# Factory class to get trainer class by name
class TrainerFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = TRAINER_NAME_TO_CLASS_MAP
     
    def get(self, trainer_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
        return super().get(trainer_name, config,
                            args_to_pass=[], kwargs_to_pass={})

# TODO: may move optimizer part to another file
OPTIMIZER_NAME_TO_CLASS_OR_INITIALIZER_MAP = {
    "Adam": Adam,
    "AdamW": AdamW
}
class OptimizerFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = OPTIMIZER_NAME_TO_CLASS_OR_INITIALIZER_MAP
    
    def get(self, optimizer_name, config=None,
                args_to_pass=[], kwargs_to_pass={}):
        return super().get(optimizer_name, config,
            args_to_pass, kwargs_to_pass)

class ExperimentPipeline(BaseExperimentPipeline):
    def __init__(self, config) -> None:
        super().__init__(config)

    def prepare_experiment(self):
        self.prepare_model()
        self.prepare_optimizer() # call this after model has been initialized
        self.prepare_scheduler()
        self.prepare_cost_function()
        self.prepare_metrics()
        self.prepare_summary_writer()
        self.prepare_dataloaders()
        self.prepare_batch_callbacks()
        self.prepare_epoch_callbacks()

        self.trainer = self.prepare_trainer()
    
    def _prepare_embeddings(self):
        # if "embedding_model_path" is in the config.. this method wil be called
        # during model preparation before loading wieghts from checkpoint
        raise NotImplementedError()
        
        


    def prepare_dataloaders(self):
        dataloader_util_class_name = self.config["dataloader_util_class_name"]
        train_batch_size = self.config["batch_size"]

        train_loader, val_loader, test_loader \
        = DataLoaderUtilFactory()\
            .get(dataloader_util_class_name, config=None)\
            .get_data_loaders(root_dir=self.config["dataloader_root_dir"],
                              batch_size=train_batch_size,
                              shuffle=self.config["shuffle"])
            
        

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def prepare_trainer(self):
        trainer_class = TrainerFactory().get_uninitialized(
            self.config["trainer_class_name"])
        
        trainer = trainer_class(model=self.model,
                    dataloader=self.train_loader,
                    cost_function=self.cost_function,
                    optimizer=self.optimizer,
                    batch_callbacks=self.batch_callbacks,
                    epoch_callbacks=self.epoch_callbacks,
                    config={
                        "num_epochs": self.config["num_epochs"]
                        }
                    )

        self.trainer = trainer
        return trainer
    

    def prepare_model(self):
        # TODO: use model config too (or make it work by creating new class)
        model = ModelFactory().get(self.config["model_class_name"])
        self.model = model
        
        # use cuda if available (TODO: decide to use config/resolve device)
        self.model.to(commonutil.resolve_device())
        
        if "embedding_model_path" in self.config:
            self._prepare_embeddings()

        if self.config["load_from_checkpoint"]:
            checkpoint_path = self.config["checkpoint_path"]
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(checkpoint_path,
                        map_location=commonutil.resolve_device())
                )
            logger.info(str(self.model))
            logger.info(f"Model Loaded")
        
        num_params = commonutil.count_number_of_params(self.model)
        num_trainable_params = commonutil.count_number_of_trainable_params(
            self.model
        )
        
        logger.info(f"Number of parameters (includes embedding): {num_params}")
        logger.info(f"Number of trainable parameters: {num_trainable_params}")
        
        return self.model
    
    def prepare_optimizer(self):
        trainable_params, trainable_param_names, frozen_params, \
                frozen_param_names = self.filter_trainer_parameters()
        logger.info(f"Frozen Parameters: {frozen_param_names}")
        logger.info(f"Trainable Parameters: {trainable_param_names} ")
        lr = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]
        # TODO: Discuss and Add subfields (2nd level nesting) in the experminet 
        # config (yaml files) to pass args and kwargs if needed 
        self.optimizer = OptimizerFactory().get(
            self.config["optimizer_class_name"],
            config=None,
            args_to_pass=[],
            kwargs_to_pass={
                "lr": lr,
                "weight_decay": weight_decay,
                "params": trainable_params
            }
        )
        logger.info(f"Using optimizer: {self.optimizer}")
    
    def prepare_scheduler(self):
        if "scheduler" not in self.config:
            return
        scheduler_name = self.config["scheduler"]
        if scheduler_name is None:
            return
        if scheduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        else:
            raise NotImplementedError()
        logger.info(f"Using scheduler: {self.scheduler}")
        
    
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
        
        self.summary_writer = SummaryWriter(
            log_dir=self.current_experiment_log_directory)

    def prepare_cost_function(self):
        class_weights = self.prepare_class_weights_for_cost_function()
        kwargs_to_pass = {}
        if class_weights is not None:
            kwargs_to_pass["weight"] = class_weights
        
        self.cost_function = CostFunctionFactory().get(
            self.config["cost_function_class_name"],
            config=None,
            args_to_pass=[],
            kwargs_to_pass=kwargs_to_pass
        )
    
    def prepare_metrics(self):
        self.metrics = {}
        self.metrics["F1"] = F1Score(threshold=self.config["threshold"])
        
    def prepare_class_weights_for_cost_function(self):
        # TODO: Add if needed
        return None
        

    def prepare_batch_callbacks(self):
        self.batch_callbacks = [self.batch_callback]

    def prepare_epoch_callbacks(self):
        self.epoch_callbacks = [self.epoch_callback]

    def run_experiment(self):
        self.trainer.train()
    
    def batch_callback(self, model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        
        if global_batch_number % self.config["batch_log_frequency"] == 0:
            logger.info(
            f"[({global_batch_number}){current_epoch}-{current_epoch_batch_number}]"
            f" Loss: {kwargs['loss']}")
        if global_batch_number % self.config["tensorboard_log_frequency"] == 0:
            self.summary_writer.add_scalar("train/loss", kwargs['loss'],
                                           global_batch_number)
    
    def epoch_callback(self, model: nn.Module, batch_data, global_batch_number,
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
            logger.exception(exc)       



class ExperimentPipelineForClassification(ExperimentPipeline):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.best_metric = None
    
    def prepare_metrics(self):
        self.metrics = {}
        self.metrics["F1"] = lambda y_pred, y_true : f1_score(
            y_true=y_true, y_pred=y_pred, average=F1_AVERAGING_SCHEME)
    
    def _prepare_embeddings(self):
        embeddings, word_to_index, index_to_word \
            = EmbeddingLoader().load(self.config["embedding_model_path"])
        
        self.model.set_embeddings(embeddings)
        self.model.set_word_to_index(word_to_index)
        self.model.set_index_to_word(index_to_word)
    
    def prepare_dataloaders(self):
        dataloader_util_class_name = self.config["dataloader_util_class_name"]
        train_batch_size = self.config["batch_size"]

        train_loader, val_loader, test_loader \
        = DataLoaderUtilFactory()\
            .get(dataloader_util_class_name, config=None)\
            .get_data_loaders(
                root_dir=self.config["dataloader_root_dir"],
                word_to_index=self.model.word_to_index, # make sure it was set
                # during model preparation
                batch_size=train_batch_size,
                shuffle=self.config["shuffle"])
            
        

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def epoch_callback(self, model: nn.Module, batch_data, global_batch_number,
     current_epoch, current_epoch_batch_number, **kwargs):
        if current_epoch == 1: # the epoch just finished
            # save the config
            self.save_config()
            with torch.no_grad():
                self.summary_writer.add_graph(
                    self.model, (batch_data[0],))# wrap data inside tuple
                #as expected by writer
    
        model.eval()
        # 

        val_f1, _ = self.compute_and_log_evaluation_metrics(
            model, current_epoch, "val")
        test_f1, _ = self.compute_and_log_evaluation_metrics(
            model, current_epoch, "test")
        if "eval_on_train_data_too" in self.config:
            if self.config["eval_on_train_data_too"]:
               _, _ = self.compute_and_log_evaluation_metrics(
                        model, current_epoch, "train/eval") 
    
        # TODO: metric can also be pulled in config
        metric_to_use_for_model_selection = val_f1 
        metric_name = "Validation F1-Score"
        
        if self.best_metric is None or \
             (self.best_metric < metric_to_use_for_model_selection):
            logger.info(f"Saving model: {metric_name} changed from "
                  f"{self.best_metric} to {metric_to_use_for_model_selection}")
            self.best_metric = metric_to_use_for_model_selection
            file_path = os.path.join(self.current_experiment_directory,
            f"best_model_{self.config['model_name_tag']}.ckpt")
            torch.save(model.state_dict(), file_path)
            self.compute_and_log_evaluation_metrics(
            model, current_epoch, "test", save_files=True)
        
        if (current_epoch % self.config["model_save_frequency"] == 0)\
            or (current_epoch == self.config["num_epochs"]):
            file_path = os.path.join(self.current_experiment_directory,
            f"model_{self.config['model_name_tag']}_"\
                +f"{str(current_epoch).zfill(4)}.ckpt")
            torch.save(model.state_dict(), file_path)

        if hasattr(self, "scheduler"):
            self.scheduler.step(metric_to_use_for_model_selection)
            next_lr = [group['lr'] for group in self.optimizer.param_groups][0]
            self.summary_writer.add_scalar("lr", next_lr,
             current_epoch)
        
        # don't forget to dump log so far
        self.summary_writer.flush()
        
        

        return self.best_metric

    def compute_and_log_evaluation_metrics(self, model, current_epoch,
        eval_type, save_files=False):
        model.eval()
        eval_loss = 0.
        n_epochs = self.config["num_epochs"]
        _loader = None
        if eval_type == "val":
            _loader = self.val_loader
        elif eval_type == "train/eval":
            _loader = self.train_loader
        elif eval_type == "test":
            _loader = self.test_loader
        else:
            raise AssertionError(f"Invalid evalutaion type: {eval_type}")
        with torch.no_grad():
            predictions = []
            targets = []

            for i, (inp, target) in enumerate(_loader):
                # device allocation to be handeled in data loader and model
                # forward pass
                pred = model.forward(inp)
                loss = self.cost_function(pred, target)
                eval_loss += loss.item()
                predictions.append(pred)
                targets.append(target)

            targets = torch.cat(targets, axis=0)
            predictions = torch.cat(predictions, axis=0)
            predictions_hard = torch.argmax(predictions, axis=1)
            f1_value = self.metrics["F1"](
                predictions_hard.to('cpu'), targets.int().to('cpu'))
            if save_files:
                target_path = os.path.join(
                    self.current_experiment_directory, f"{eval_type}_true.npz")
                pred_path = os.path.join(
                    self.current_experiment_directory, f"{eval_type}_pred.npz"
                )                           
                np.save(target_path, targets.cpu().detach().numpy())
                np.save(pred_path, predictions_hard.cpu().detach().numpy())
                logger.info(f"Saved {eval_type} files at:\ni) {target_path}"
                            f"\nii) {pred_path}")

        self.summary_writer.add_scalar(
            f"{eval_type}/loss", eval_loss / len(_loader.dataset),
            current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/F1", f1_value,
                                       current_epoch)
        logger.info(f"{eval_type} loss after epoch {current_epoch}/{n_epochs}:"
                    f" {eval_loss / len(_loader.dataset)}")
        logger.info(
            f"{eval_type} F1-Score after epoch"
            f" {current_epoch}/{n_epochs}: {f1_value}")
        
        return f1_value, loss

        

class EvaluationPipelineForClassification(ExperimentPipelineForClassification):
    def __init__(self, config) -> None:
        super().__init__(config)
        
    def prepare_experiment(self):
        return super().prepare_experiment()
    
    def prepare_summary_writer(self):
        # return mock summary writer
        class _DummyWriter:
            def add_scalar(*args, **kwargs):
                return
            
            def flush(*args, **kwargs):
                return
            
            def add_graph(*args, **kwargs):
                return
        self.summary_writer = _DummyWriter()
        
        return self.summary_writer
    
    def run_experiment(self):
        model = self.model
        experiment_tag = self.config["experiment_metadata"]["tag"]
        timestamp = get_timestamp_str()
        self.current_experiment_directory = os.path.join(
            self.config["logdir"],timestamp+"EVALUATION_"+experiment_tag)
        os.makedirs(self.current_experiment_directory, exist_ok=True)
        _ = self.compute_and_log_evaluation_metrics(
            model, 0, "val", save_files=True)
        logger.info(f"Validation accuracy= {self.get_accuracy('val')}")
        _ = self.compute_and_log_evaluation_metrics(
            model, 0, "test", save_files=True)
        logger.info(f"Test accuracy= {self.get_accuracy('test')}")
    
    def get_accuracy(self, eval_type):
        """Assumes prediction files are saved already
        
        """
        target_path = os.path.join(
                    self.current_experiment_directory, f"{eval_type}_true.npz.npy")
        pred_path = os.path.join(
            self.current_experiment_directory, f"{eval_type}_pred.npz.npy"
        )
        
        acc = accuracy_score(y_true=np.load(target_path),
                             y_pred=np.load(pred_path))
        
        return acc
        
    
    
        
        

        
PIPELINE_NAME_TO_CLASS_MAP = {
    "ExperimentPipeline": ExperimentPipeline,
    "ExperimentPipelineForClassification": ExperimentPipelineForClassification,
    "EvaluationPipelineForClassification": EvaluationPipelineForClassification
}


def main():
    from util import PROJECTPATH
    from pathlib import Path
    DEFAULT_CONFIG_LOCATION \
        = str(Path(PROJECTPATH)/"src/experiment_configs/exp_04_task2_cnn_res.yaml")
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str,
                            default=DEFAULT_CONFIG_LOCATION)
    args = argparser.parse_args()
    
    config_data = None
    with open(args.config, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    

    pipeline_class = PIPELINE_NAME_TO_CLASS_MAP[ config_data["pipeline_class"]]
    pipeline = pipeline_class(config=config_data)
    pipeline.prepare_experiment()
    pipeline.run_experiment()

if __name__ == "__main__":
    main()


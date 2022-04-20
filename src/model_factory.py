import torch
from util import BaseFactory
from model_cnn_res import CnnWithResidualConnection
from model_ann import (
    FullyConnectedModel, FullyConnectedModelWithDropout,
    FullyConnectedModelUnfrozenEmdeddings, FullyConnectedModel300)
# model mapping
MODEL_NAME_TO_CLASS_OR_INTIALIZER_MAP = {
    "CnnWithResidualConnection": CnnWithResidualConnection,
    "FullyConnectedModel": FullyConnectedModel,
    "FullyConnectedModelWithDropout": FullyConnectedModelWithDropout,
    "FullyConnectedModelUnfrozenEmdeddings": \
        FullyConnectedModelUnfrozenEmdeddings,
    "FullyConnectedModel300": FullyConnectedModel300
}

# For saved models 
MODEL_NAME_TO_WEIGHTS_PATH = {

}
class ModelFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = MODEL_NAME_TO_CLASS_OR_INTIALIZER_MAP

    def get(self, model_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
        # handle models 
        return super().get(model_name, config,
                            args_to_pass, kwargs_to_pass)

class TrainedModelFactory(ModelFactory):
    def __init__(self, config = {}) -> None:
        super().__init__()
        # if config has `model_name_to_weights_path`
        self.config = config
        if "model_name_to_weights_path" not in self.config:
            self.config["model_name_to_weights_path"] \
                = MODEL_NAME_TO_WEIGHTS_PATH

        self.model_weights_path = self.config["model_name_to_weights_path"]
    
    def get(self, model_name, config=None):
        # TODO: config not being used currently
        model_class =  super().get(model_name)
        model_weights_path = self.model_weights_path[model_name]

        model: torch.nn.Module = model_class() # Assumes model does not need init params

        state_dict = torch.load(model_weights_path)
        if hasattr(model, "load_state_dict_for_eval"):
            model.load_state_dict_for_eval(state_dict=state_dict, strict=True)
        else:
            model.load_state_dict(state_dict=state_dict, strict=True)
        # make sure to call model.eval() or model.train() based on the usage
        return model

    def get_lazy_loader(self, model_name):
        return lambda : self.get(model_name)
    
    def load_from_location(self, model_name, model_weights_path):
        model_class =  super().get(model_name)

        model: torch.nn.Module = model_class() # Assumes model does not need init params

        state_dict = torch.load(model_weights_path)
        if hasattr(model, "load_state_dict_for_eval"):
            model.load_state_dict_for_eval(state_dict=state_dict, strict=True)
        else:
            model.load_state_dict(state_dict=state_dict, strict=True)
        # make sure to call model.eval() or model.train() based on the usage
        return model
        

if __name__ == "__main__":
    # >>> model_factory = TrainedModelFactory()
    model_factory = ModelFactory()
    model = model_factory.get("PrunedResnet50")
    print(model)
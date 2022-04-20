import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.functional
from util import resolve_device, BaseFactory
from torch.nn import CrossEntropyLoss


COST_FUNCTION_NAME_TO_CLASS_MAP = {
    "CrossEntropyLoss": CrossEntropyLoss
}

class CostFunctionFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = COST_FUNCTION_NAME_TO_CLASS_MAP
    
    def get(self, cost_function_class_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
        return super().get(cost_function_class_name, config,
                           args_to_pass, kwargs_to_pass)
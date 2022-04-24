import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.functional
from util import resolve_device, BaseFactory
from torch.nn import CrossEntropyLoss
import numpy as np

CLASS_FREQ = torch.tensor(data=np.array([21727, 27168, 59353, 13839, 57953]),
                          dtype=torch.float32, requires_grad=False)
CLASS_FREQ = CLASS_FREQ/torch.max(CLASS_FREQ) # scale b/w 0 to 1.0
WT_INV = 1/CLASS_FREQ
WT_INV_SQR = 1/(CLASS_FREQ*CLASS_FREQ)


COST_FUNCTION_NAME_TO_CLASS_MAP = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "WeightedCrossEntropyLoss": lambda : CrossEntropyLoss(weight=WT_INV),
    "WeightedCrossEntropyLossInvSqr": lambda : CrossEntropyLoss(
        weight=WT_INV_SQR.to(resolve_device()))
}

class CostFunctionFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = COST_FUNCTION_NAME_TO_CLASS_MAP
    
    def get(self, cost_function_class_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
        return super().get(cost_function_class_name, config,
                           args_to_pass, kwargs_to_pass)
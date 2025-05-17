from easydict import EasyDict

from utils.enums import LayerType, WeightsInitType

model_cfg = EasyDict()

# Layers configuration
model_cfg.layers = [
    {'type': LayerType.Linear, 'params': {'in_features': 961, 'out_features': 2048, 'bias': True}},
    {'type': LayerType.ReLU, 'params': {}},
    {'type': LayerType.Dropout, 'params': {'p': 0.5}},
    {'type': LayerType.Linear, 'params': {'in_features': 2048, 'out_features': 256, 'bias': True}},
    {'type': LayerType.ReLU, 'params': {}},
    {'type': LayerType.Dropout, 'params': {'p': 0.5}},
    {'type': LayerType.Linear, 'params': {'in_features': 256, 'out_features': 3, 'bias': True}},
]
# Weights and bias initialization
model_cfg.params = EasyDict()
# model_cfg.params.init_type = WeightsInitType.normal
model_cfg.params.init_type = WeightsInitType.xavier_uniform
# More details about initialization methods parameters can be found here: https://pytorch.org/docs/stable/nn.init.html
model_cfg.params.init_kwargs = {}
model_cfg.params.zero_bias = True

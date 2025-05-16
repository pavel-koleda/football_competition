from easydict import EasyDict

from utils.enums import WeightsInitType, LayerType

model_cfg = EasyDict()

# Layers configuration
model_cfg.layers = [
    {'type': LayerType.Linear, 'params': {'in_features': 32 * 32, 'out_features': 128, 'bias': True}},
    {'type': LayerType.ReLU, 'params': {}},
    {'type': LayerType.Dropout, 'params': {'p': 0.2}},
    {'type': LayerType.Linear, 'params': {'in_features': 128, 'out_features': 64, 'bias': True}},
    {'type': LayerType.ReLU, 'params': {}},
    {'type': LayerType.Dropout, 'params': {'p': 0.2}},
    {'type': LayerType.Linear, 'params': {'in_features': 64, 'out_features': 7, 'bias': True}},
]
# Weights and bias initialization
model_cfg.params = EasyDict()
model_cfg.params.init_type = WeightsInitType.xavier_uniform
# More details about initialization methods parameters can be found here: https://pytorch.org/docs/stable/nn.init.html
model_cfg.params.init_kwargs = {}
model_cfg.params.zero_bias = True

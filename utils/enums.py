from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'validation', 'test'))
PreprocessingType = IntEnum('PreprocessingType', ('normalization', 'standardization', 'identical'))
LayerType = IntEnum('LayerType', ('Linear', 'ReLU', 'Dropout'))
WeightsInitType = IntEnum(
    'WeightsInitType', ('normal', 'uniform', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
)

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.layers = self._init_layers()

        self.init_function = getattr(nn.init, self.config.params.init_type.name + '_')
        self.apply(self._init_weights)

    def _init_layers(self) -> nn.Sequential:
        """MLP layers initialization.

        Returns:
            nn.Sequential or nn.ModuleList with all the initialized layers
        """
        return nn.Sequential(*[getattr(nn, layer['type'].name)(**layer['params']) for layer in self.config.layers])

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        """Layer parameters initialization.

        Args:
            module: The model layer.
        """
        if isinstance(module, nn.Linear):
            self.init_function(module.weight, **self.config.params.init_kwargs)

            if module.bias is not None:
                if self.config.params.zero_bias or not self.init_function.__name__.startswith(('_normal', '_uniform')):
                    nn.init.zeros_(module.bias)
                else:
                    self.init_function(module.bias, **self.config.params.init_kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward propagation implementation.

        This method propagates inputs through all the layers from self.layers.

        Args:
            inputs: The inputs tensor of shape (batch_size, height, width).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, classes_num).
        """
        inputs = inputs.view(inputs.size(0), -1)
        return self.layers(inputs)

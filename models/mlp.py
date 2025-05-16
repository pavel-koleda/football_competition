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
        # TODO: Make a sequence of layers using torch.nn modules and nn.Sequential:
        #       - Make a list of layers initialized using getattr(nn, layer['type'].name)(**layer['params']),
        #               for each layer from self.config.layers.
        #       - Initialize torch.nn.Sequential class providing layers list items as positional arguments
        #               (torch.nn.Sequential(layer_1, layer_2,...))
        raise NotImplementedError

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        """Layer parameters initialization.

        Args:
            module: The model layer.
        """
        #  TODO: Implement parameters initialization:
        #        - Check if module is of nn.Linear type using isinstance(module, nn.Linear)
        #        - if module is nn.Linear:
        #               1. Call self.init_function to initialize weights: provide module.weight as positional argument
        #                      and params from self.config.params.init_kwargs as keyword arguments
        #               2. Initialize bias if module.bias is not None:
        #                      - Use nn.init.zeros_() method if self.config.params.zero_bias is True
        #                           or init method is either Xavier or He (kaiming)
        #                      - Otherwise call self.init_function(), providing module.bias
        #                           and self.config.params.init_kwargs in the same way as for the module.weight
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward propagation implementation.

        This method propagates inputs through all the layers from self.layers.

        Args:
            inputs: The inputs tensor of shape (batch_size, height, width).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, classes_num).
        """
        # TODO: Implement forward pass:
        #       1. Reshape inputs into a 2D tensor (first dimension is the mini-batch size) using inputs.view() method
        #       2. Pass reshaped inputs through self.layers and return the result
        raise NotImplementedError

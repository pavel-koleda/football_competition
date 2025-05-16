import os
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from dataloaders.batch_samplers import Upsampling
from dataset.emotions_dataset import EmotionsDataset
from models.mlp import MLP
from utils.common_functions import set_seed
from utils.enums import SetType
from utils.logger import NeptuneLogger
from utils.metrics import balanced_accuracy_score, confusion_matrix
from utils.visualization import plot_confusion_matrix


class Trainer:
    """A class for model training."""

    def __init__(self, config, init_logger=True):
        self.config = config
        set_seed(self.config.seed)

        self._prepare_data()
        self._prepare_model()

        self._init_logger(init_logger)

    def _init_logger(self, init_logger: bool):
        if init_logger:
            self.logger = NeptuneLogger(self.config.neptune)
            if not self.config.train.continue_train:
                self.logger.log_hyperparameters(self.config)

    def _prepare_data(self):
        """Prepares training and validation data."""
        data_cfg = self.config.data
        batch_size = self.config.train.batch_size

        train_transforms = data_cfg.train_transforms
        validation_transforms = data_cfg.eval_transforms

        self.train_dataset = EmotionsDataset(data_cfg, SetType.train, transforms=train_transforms)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size, drop_last=True, sampler=Upsampling(self.train_dataset),
            num_workers=os.cpu_count(),
        )
        self.eval_train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=False)

        self.validation_dataset = EmotionsDataset(data_cfg, SetType.validation, transforms=validation_transforms)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)

    def _prepare_model(self):
        """Prepares model, optimizer and loss function."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLP(self.config.model).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.train.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def save(self, filepath: str):
        """Saves trained model."""
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            os.path.join(self.config.checkpoints_dir, filepath)
        )

    def load(self, filepath: str):
        """Loads trained model."""
        checkpoint = torch.load(os.path.join(self.config.checkpoints_dir, filepath), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def update_best_params(self, valid_metric: float, best_metric: float) -> float:
        """Update best parameters: saves model if metrics exceeds the best values achieved."""
        if best_metric < valid_metric:
            self.save(self.config.best_checkpoint_name)
            best_metric = valid_metric
        return best_metric

    def make_step(self, batch: dict, update_model=False) -> (float, np.ndarray):
        """This method performs one step, including forward pass, calculation of the target function, backward
        pass and updating the model weights (if update_model is True).

        Args:
            batch: A batch data dictionary.
            update_model: If True it is necessary to perform a backward pass and update the model weights.

        Returns:
            loss: The loss function value.
            output: The model output (batch_size x classes_num).
        """
        # TODO: Implement one step of forward and backward propagation:
        #       1. Get images and targets from batch, "move" them to self.device
        #       2. Get model output by passing batch images through the model instance
        #       3. Compute loss with self.criterion using batch targets and obtained model output
        #       4. If update_model parameter is True, make backward propagation:
        #           - Set the gradient of the layer parameters to zero using self.optimizer.zero_grad()
        #           - Compute the gradient of the computed loss using its backward() method
        #           - Update the model parameters using self.optimizer.step()
        raise NotImplementedError

    def train_epoch(self):
        """Trains the model on training data for one epoch.

        The method goes through all train_dataloader batches and calls the self.make_step() method at each step.
        """
        self.model.train()
        pbar = tqdm(self.train_dataloader)
        # TODO: Implement the training process for one epoch. For all batches in train_dataloader (pbar) do:
        #       1. Make training step by calling self.make_step() method on a batch with update_model=True
        #       2. Get the model predictions from the outputs using argmax() method with correct axis
        #       3. Compute metrics using batch targets and obtained model predictions
        #       4. Log the calculated loss value and metrics with self.logger.save_metrics() and correct SetType name
        #       5. (Optional) Update tqdm progress bar state (e. g. with pbar.set_description() method)
        #               with current loss and metric values
        raise NotImplementedError

    def fit(self):
        """The main model training loop."""
        best_metric = 0
        start_epoch = 0
        # TODO: Implement the main model training loop iterating over the epochs. First, check whether it is necessary
        #           to continue the previous training experiment using self.config.train.continue_train parameter. If
        #           the condition is met, get the epoch where the experiment was stopped from
        #           self.config.train.checkpoint_from_epoch. Load the model from this epoch using self.load() method,
        #           and set start_epoch to the obtained epoch + 1
        #       Then at each epoch starting from the start_epoch and ending at self.config.num_epochs:
        #           1. The model is first trained on the training data using the self.train_epoch() method
        #           2. The model performance is then evaluated on the train (self.eval_train_dataloader) and validation
        #                   (self.validation_dataloader) data with self.evaluate() method
        #           3. The model is saved if needed (check self.config.checkpoint_save_frequency)
        #           4. If performance metrics on the validation data exceeds the best values achieved,
        #                   model parameters should be saved with save() method (call self.update_best_params() method
        #                   and update best_metric value)
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, epoch: int, dataloader: DataLoader, set_type: SetType) -> float:
        """Evaluation.

        The method is used to make the model performance evaluation on training/validation/test data.

        Args:
            epoch: A current training epoch.
            dataloader: The dataloader for the chosen set type.
            set_type: The set type chosen to evaluate.
        """
        self.model.eval()

        total_loss = []
        all_outputs, all_labels = [], []
        # TODO: To implement the model performance evaluation for each batch in the given dataloader do:
        #       1. Make model forward pass using self.make_step(batch, update_model=False)
        #       2. Add loss value to total_loss list
        #       3. Add model output to all_outputs list
        #       4. Add batch targets to all_labels list
        #    Get total loss (by averaging gathered losses) and metrics values (using gathered outputs to get
        #           predictions with argmax), log them with logger. You can also make CM using plot_confusion_matrix()
        #           method and log it using self.logger.save_plot. Return calculated metric value
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, model_path: str, dataloader: DataLoader) -> (list, list):
        """Gets model predictions for a given dataloader."""
        self.load(model_path)
        self.model.eval()
        all_outputs, all_image_paths = [], []
        # TODO: To implement this method, for each batch in the given dataloader do:
        #       1. Add batch image paths to all_image_paths list
        #       2. Get the model output for the batch images (don't forget to move batch images to self.device)
        #       3. Add model output to all_outputs list
        #       Get predictions using gathered model outputs and argmax method and return concatenated predictions
        #               and image paths as a result
        raise NotImplementedError

    def batch_overfit(self):
        """One batch overfitting.

        This feature can be useful for debugging and evaluating your model's ability to learn and update its weights.
        """
        self.model.train()
        batch = next(iter(self.train_dataloader))
        # TODO: To implement this method, for each iteration from 0 to self.config.overfit.num_iterations:
        #       1. Make model forward pass using self.make_step(batch, update_model=True)
        #       2. Get metric value using batch targets and predictions from the obtained model output
        #       3. Log calculated loss and metric values via self.logger.save_metrics
        raise NotImplementedError

import numpy as np
from torch.utils.data import Sampler


class Upsampling(Sampler):
    """Upsampling Minority Class.

    Upsampling is a technique used to create additional data points of the minority class to balance class labels.
        This is usually done by duplicating existing samples or creating new ones
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.indices = np.arange(len(dataset))
        self.labels = dataset.labels

        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.max_count = max(counts)
        self.class_indices = {label: np.where(self.labels == label)[0] for label in unique_labels}

    def __iter__(self):
        indices = []

        # TODO: To implement this method:
        #       1. For each class in self.class_indices:
        #           - Add all indices of the class elements to the indices list
        #           - If the number of samples is less than the maximum number, randomly sample element indices
        #               from this class so that the total number of indices equals the maximum number,
        #               add sampled indices to the indices list
        #       2. Shuffle indices (e.g. using np.random.permutation)
        #       3. Return iterator of the gathered indices list (e.g. using iter method)
        raise NotImplementedError

    def __len__(self):
        return self.max_count * len(self.class_indices)

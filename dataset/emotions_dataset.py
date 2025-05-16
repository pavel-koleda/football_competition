import os

from PIL import Image
from torch.utils.data import Dataset

from utils.common_functions import read_dataframe_file
from utils.enums import SetType


class EmotionsDataset(Dataset):
    """A class for the Emotions dataset. This class defines how data is loaded."""

    def __init__(self, config, set_type: SetType, transforms=None):
        self.config = config
        self.set_type = set_type
        self.transforms = transforms

        # Reading an annotation file that contains the image path, set_type, and target values for the entire dataset
        annotation = read_dataframe_file(os.path.join(config.path_to_data, config.annot_filename))

        # Filter the annotation file according to set_type
        self.annotation = annotation[annotation['set'] == self.set_type.name]

        # TODO: Get image paths from annotation dataframe's 'path' column
        self._paths = ...
        # TODO: Make mapping from annotation dataframe's 'target' column to int values using self.config.label_mapping.
        #       When set_type is SetType.test, the target does not exist.
        self._targets = ...

    @property
    def labels(self):
        return self._targets

    def __len__(self):
        # TODO: Return the number of samples in the dataset
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Loads and returns one sample from a dataset with the given idx index.

        Returns:
            A dict with the following data:
                {
                    'image: image (numpy.ndarray),
                    'target': target (int),
                    'path': image path (str)
                }
        """
        # TODO: To implement the method:
        #       1. Read the image with the provided index from the dataset by its path using PIL,
        #               convert it to GRAYSCALE mode
        #       2. Call the self.transforms functions for the image (if self.transforms is set up)
        #       3. Return the image, the corresponding target and image path as a dictionary
        #                with keys "image", "target", "path", respectively
        raise NotImplementedError

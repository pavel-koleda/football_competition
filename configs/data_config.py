import os

from easydict import EasyDict
from torchvision import transforms

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

data_cfg = EasyDict()

# Path to the directory with dataset files
data_cfg.path_to_data = os.path.join(ROOT_DIR, 'data', 'emotion_detection')
data_cfg.annot_filename = 'data_info.csv'

# Label mapping
data_cfg.label_mapping = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}
data_cfg.classes_num = 7

# Training configuration
data_cfg.train_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
data_cfg.eval_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

import os

from easydict import EasyDict

from configs.data_config import data_cfg
from configs.model_config import model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

experiment_cfg = EasyDict()
experiment_cfg.seed = 0
experiment_cfg.num_epochs = 1000

# Train parameters
experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 2048*2
experiment_cfg.train.learning_rate = 1e-3
experiment_cfg.train.continue_train = False
experiment_cfg.train.checkpoint_from_epoch = None

# Overfit parameters
experiment_cfg.overfit = EasyDict()
experiment_cfg.overfit.num_iterations = 500

# Neptune parameters
experiment_cfg.neptune = EasyDict()
experiment_cfg.neptune.env_path = os.path.join(ROOT_DIR, '.env')
experiment_cfg.neptune.project = 'Emotions-detection/emotion-detection'
experiment_cfg.neptune.experiment_name = 'football_prediction_run_1'
experiment_cfg.neptune.run_id = None #None #'EM-65'
experiment_cfg.neptune.dependencies_path = os.path.join(ROOT_DIR, 'requirements.txt')


# Neptune parameters
experiment_cfg.neptune = EasyDict()
experiment_cfg.neptune.env_path = os.path.join(ROOT_DIR, '.env')
experiment_cfg.neptune.project = 'Emotions-detection/emotion-detection'
experiment_cfg.neptune.experiment_name = ''
experiment_cfg.neptune.run_id = None
experiment_cfg.neptune.dependencies_path = os.path.join(ROOT_DIR, 'requirements.txt')

# MLflow parameters
experiment_cfg.mlflow = EasyDict()
experiment_cfg.mlflow.tracking_uri = 'http://127.0.0.1:5000'
experiment_cfg.mlflow.experiment_name = ''
experiment_cfg.mlflow.run_id = None
experiment_cfg.mlflow.dependencies_path = os.path.join(ROOT_DIR, 'requirements.txt')

# Checkpoints parameters
experiment_cfg.checkpoints_dir = os.path.join(
    ROOT_DIR, 'experiments', experiment_cfg.neptune.experiment_name, 'checkpoints'
)
experiment_cfg.checkpoint_save_frequency = 1
experiment_cfg.checkpoint_name = 'checkpoint_%s'
experiment_cfg.best_checkpoint_name = 'best_checkpoint'

# Data parameters
experiment_cfg.data = data_cfg

# Model parameters
experiment_cfg.model = model_cfg

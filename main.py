import pandas as pd
from torch.utils.data import DataLoader

from configs.data_config import data_cfg
from configs.experiment_config import experiment_cfg
from dataset.emotions_dataset import EmotionsDataset
from executors.trainer import Trainer
from utils.enums import SetType


def train():
    trainer = Trainer(experiment_cfg)

    # One batch overfitting
    # trainer.batch_overfit()

    # Model training
    trainer.fit()


def predict():
    trainer = Trainer(experiment_cfg, init_logger=False)

    # Get data to make predictions on
    test_dataset = EmotionsDataset(data_cfg, SetType.test, transforms=data_cfg.eval_transforms)
    test_dataloader = DataLoader(test_dataset, experiment_cfg.train.batch_size, shuffle=False)

    # Get predictions
    model_path = experiment_cfg.best_checkpoint_name
    predictions, image_paths = trainer.predict(model_path, test_dataloader)

    # Save results to submission file
    test_results_df = pd.DataFrame({'ID': image_paths, 'prediction': predictions})
    test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    train()
    # predict()

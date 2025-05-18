import pandas as pd
from torch.utils.data import DataLoader

from configs.data_config import data_cfg
from configs.experiment_config import experiment_cfg
from dataset.football_dataset import FootballDataset
from executors.trainer import Trainer
from utils.common_functions import read_dataframe_file
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
    test_dataset = FootballDataset(data_cfg, SetType.test, transforms=None)
    test_dataloader = DataLoader(test_dataset, experiment_cfg.train.batch_size, shuffle=False)

    # Get predictions
    model_path = experiment_cfg.best_checkpoint_name
    predictions = trainer.predict(model_path, test_dataloader)
    matches_test = read_dataframe_file(r'data\Football Dataset\matches_test.pickle')
    # Save results to submission file
    test_results_df = pd.DataFrame({'id': matches_test.index.to_list(), 'match_result': predictions})
    mappings = {str(value): key for key, value in data_cfg.label_mapping.items()}

    test_results_df['match_result'] = test_results_df['match_result'].astype(str)

    test_results_df.loc[:, 'match_result'] = test_results_df['match_result'].map(mappings)
    test_results_df.to_csv('test_predictions.csv', index=False)





if __name__ == '__main__':
    # train()
    predict()

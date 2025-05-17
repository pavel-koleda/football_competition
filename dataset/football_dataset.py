import os
import pandas as pd

from torch.utils.data import Dataset

from utils.common_functions import read_dataframe_file
from utils.enums import SetType


class FootballDataset(Dataset):
    """A class for the Football dataset. This class defines how data is loaded."""

    def __init__(self, config, set_type: SetType, transforms=None):
        
        self.config = config
        self.set_type = set_type
        self.transforms = transforms

        matches = read_dataframe_file(os.path.join(config.path_to_data, f'matches_{self.set_type.name}.pickle'))
        players = read_dataframe_file(os.path.join(config.path_to_data, 'players.pickle'))
        teams = read_dataframe_file(os.path.join(config.path_to_data, 'teams.pickle'))

        self.fill_matches_missing_values(matches)
        self.ohe_features_encode(self.config.categories, matches, players, teams)

        self.drop_matches_columns(matches)
        
        self.add_season_start_end_index(matches, players, teams)
        
        matches = self.add_players_info(matches, players)
        self.add_teams_info(matches, teams)
        
        if self.set_type.name != 'test':
            self._targets = self.matches['match_result'].map(self.config.label_mapping).tolist() 

        self._inputs = matches.to_numpy()


    def fill_matches_missing_values(self, matches):
        #TODO: Заполнить стадион через k ближайших соседей вместо drop
        matches.drop(columns=['stadium_name'], inplace=True)

        matches['game week'] = matches['game week'].fillna(matches['game week'].mean())


    def ohe_features_encode(self, categories: dict, *args: pd.DataFrame):
        for df in args:
            ohe_columns = []
            
            for category in categories:
                if category in df.columns:
                    df[category] = pd.Categorical(df[category], categories=categories[category], ordered=True)
                    ohe_columns.append(category)
                    
            if len(ohe_columns) > 0:
                dummies = pd.get_dummies(df, columns=ohe_columns)
                df[dummies.columns] = dummies
                df.drop(columns=ohe_columns, inplace=True)


    def add_season_start_end_index(self, *args):
        for df in args:
            df['season_start'] = df['season'].apply(lambda x: int(x.split('/')[0]))
            df['season_end'] = df['season'].apply(lambda x: int(x.split('/')[1]))
            df.drop(columns='season', inplace=True)


    def drop_matches_columns(self, matches):
        matches.drop(columns=['home_team_name', 
                              'away_team_name', 
                              'home_team_bench_players', 
                              'away_team_bench_players', 
                              'attendance', 
                              'referee',
                              'timestamp'], inplace=True)

        if self.set_type.name != 'test':
            matches.drop(columns=['home_team_goal_count',
                                  'away_team_goal_count',
                                  'total_goal_count',
                                  'total_goals_at_half_time'], inplace=True)


    def add_players_info(self, matches, players) -> pd.DataFrame:
        #заменяем id команды информацией о ней
        raise NotImplementedError


    def add_teams_info(self, matches, players):
        #заменяем id команды информацией о ней
        raise NotImplementedError


    @property
    def labels(self):
        return self._targets


    def __len__(self):
        return len(self._inputs)


    def __getitem__(self, idx: int) -> dict:
        """Loads and returns one sample from a dataset with the given idx index.

        Returns:
            A dict with the following data:
                {
                    'input: match data record,
                    'target': target (int)
                }
        """
        input = self.inputs[idx]

        if self.transforms is not None:
            input = self.transforms(input)

        return {'input': input, 'target': self._targets[idx]}

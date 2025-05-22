import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.common_functions import read_dataframe_file
from utils.enums import SetType
from utils.preprocessing import Preprocessing


class FootballDataset(Dataset):
    """A class for the Football dataset. This class defines how data is loaded."""

    def __init__(self, config, set_type: SetType, transforms=None):
        
        self.config = config
        self.set_type = set_type
        self.transforms = transforms

        # Preprocessing class initialization
        self.preprocessing = Preprocessing(config.preprocess_type)

        matches = read_dataframe_file(os.path.join(config.path_to_data, f'matches_{self.set_type.name}.pickle'))
        players = read_dataframe_file(os.path.join(config.path_to_data, 'players.pickle'))
        teams = read_dataframe_file(os.path.join(config.path_to_data, 'teams.pickle'))

        self.fill_matches_missing_values(matches)

        players = self.drop_players_columns(players)
        players['birthday_year'] = players['birthday_gmt'].apply(lambda x: int(x.split('/')[0]))
        players = players.drop(columns=['birthday_gmt'])

        matches = self.drop_matches_columns(matches)
        matches = self.drop_matches_2013_season(matches)
        
        self.add_season_start_end_index(matches, players, teams)
        matches = self.add_players_info(matches, players)
        matches = self.add_teams_info(matches, teams)
        
        matches = self.ohe_features_encode(self.config.categories, matches)

        # matches = self.ohe_features_encode({'referee': matches['referee'].unique()}, matches)

        matches = self.add_date_features(matches)

        # Преобразование категориальных столбцов в числовые
        for col in matches.select_dtypes(['category']).columns:
            matches[col] = matches[col].cat.codes

        if self.set_type.name != 'test':
            self._targets = matches['match_result'].map(self.config.label_mapping).to_numpy()
            matches = matches.drop(columns=['match_result'])

        matches = matches.fillna(0)
        
        print(matches.shape)
        
        # Подсчет столбцов с префиксами "home_team_" и "away_team_"
        home_team_columns = [col for col in matches.columns if col.startswith('home_team_')]
        away_team_columns = [col for col in matches.columns if col.startswith('away_team_')]

        print(f"Количество столбцов с префиксом 'home_team_': {len(home_team_columns)}")
        print(f"Количество столбцов с префиксом 'away_team_': {len(away_team_columns)}")


        # # Вычитание значений столбцов "away_team_" из соответствующих столбцов "home_team_"
        # for home_col in home_team_columns:
        #     away_col = home_col.replace('home_team_', 'away_team_')
        #     if away_col in matches.columns:
        #         matches[home_col] = matches[home_col] - matches[away_col]

        # # Удаление столбцов "away_team_"
        # matches = matches.drop(columns=away_team_columns)

        print(matches.shape)

        matches = matches.to_numpy(dtype='float')

        matches = self.preprocessing.train(matches)

        self._inputs = matches


    def fill_matches_missing_values(self, matches):
        #TODO: Заполнить стадион через k ближайших соседей вместо drop
        matches.drop(columns=['stadium_name'], inplace=True)

        matches['game week'] = matches['game week'].fillna(matches['game week'].mean())


    def ohe_features_encode(self, categories: dict, df: pd.DataFrame) -> pd.DataFrame:
        
        for category in categories:
            matching_features = [col for col in df.columns if col.startswith(category)]
            if len(matching_features) > 0:
                for feature in matching_features:
                    df[feature] = pd.Categorical(df[feature], categories=categories[category], ordered=True)
                df = pd.get_dummies(df, columns=matching_features)

        return df


    def add_season_start_end_index(self, *args):
        for df in args:
            df['season_start'] = df['season'].apply(lambda x: int(x.split('/')[0]))
            df['season_end'] = df['season'].apply(lambda x: int(x.split('/')[1]))
            df.drop(columns='season', inplace=True)


    def drop_matches_columns(self, matches) -> pd.DataFrame:
        matches = matches.drop(columns=['home_team_name', 
                                        'away_team_name', 
                                        'home_team_bench_players', 
                                        'away_team_bench_players', 
                                        # 'attendance', 
                                        # 'referee',
                                        'timestamp'])

        if self.set_type.name != 'test':
            matches = matches.drop(columns=['home_team_goal_count',
                                            'away_team_goal_count',
                                            'total_goal_count',
                                            'total_goals_at_half_time'])
            
        return matches

    def drop_players_columns(self, players) -> pd.DataFrame:
        return players.drop(columns=['full_name',
                                     'team',
                                     'code',
                                     'nationality'])


    def drop_matches_2013_season(self, matches) -> pd.DataFrame:
        return matches.loc[matches['season'] != '2013/2014']

    
    def add_players_info(self, matches, players: pd.DataFrame) -> pd.DataFrame:
        """Получаю статистику игрока из предыдущего сезона
           Для этого создаю отдельные колонки с идентификаторами игроков
           Затем мержу данные из второй таблицы по условию совпадения идентификатора, лиги, начало текущего сезона = окончание предыдущего"""
        
        player_positions = dict(players.drop_duplicates(subset='id')[['id', 'position']].values)

        home_players_columns = matches['home_team_main_players'].apply(pd.Series)
        home_players_columns.columns = ["home_player_" + str(i+1) for i in range(home_players_columns.shape[1])]
        result_df = pd.concat([matches.drop(columns=['home_team_main_players']), home_players_columns], axis=1)

        away_players_columns = result_df['away_team_main_players'].apply(pd.Series)
        away_players_columns.columns = ["away_player_" + str(i+1) for i in range(away_players_columns.shape[1])]
        result_df = pd.concat([result_df.drop(columns=['away_team_main_players']), away_players_columns], axis=1)

        players = players.drop(columns=['season_start'])

        for i in range(11):
            result_df = result_df.merge(players, 
                                        how='left', 
                                        left_on=[f'home_player_{i+1}', 'season_start', 'home_team_id', 'league'], 
                                        right_on=['id', 'season_end', 'team_id', 'league'],
                                        suffixes=('',f'_home_player{i+1}'))
            

            #Для первой записи суффикс в поле position будет отсутствовать, поэтому так
            position_field = 'position' if i == 0 else f'position_home_player{i + 1}'
            result_df[position_field] = result_df[position_field].fillna(result_df[f'home_player_{i+1}'].map(player_positions))                
            result_df = result_df.drop(columns=[f'season_end_home_player{i+1}', f'id_home_player{i+1}'])

        for i in range(11):
            result_df = result_df.merge(players, 
                            how='left', 
                            left_on=[f'away_player_{i+1}', 'season_start', 'away_team_id', 'league'], 
                            right_on=['id', 'season_end', 'team_id', 'league'],
                            suffixes=('',f'_away_player{i+1}'))
            
            result_df[f'position_away_player{i+1}'] = result_df[f'position_away_player{i+1}'].fillna(result_df[f'away_player_{i+1}'].map(player_positions)) 
            result_df = result_df.drop(columns=[f'season_end_away_player{i+1}', f'id_away_player{i+1}'])

        #TODO: нужно заполнить либо данными команды (но есть примеры, когда информация о всей команде в игроках отсутствует), либо чем-то более осмысленным
        #NaN также может быть из-за ошибки в данных, например в players не заполнен id команды, но таких всего около 80 записей

        numerical_features = result_df.select_dtypes(include=['number'])
        mean_values = numerical_features.mean()
        result_df[numerical_features.columns] = result_df[numerical_features.columns].fillna(mean_values)

        return result_df
        pd.mer

    def add_teams_info(self, matches, teams) -> pd.DataFrame:
        #заменяем id команды информацией о ней
        home_teams = teams.copy()
        home_teams = home_teams.add_prefix('home_team_')

        result_df = matches.merge(home_teams, 
                    how='left', 
                    left_on=['home_team_id', 'season_start', 'league'], 
                    right_on=['home_team_id', 'home_team_season_end', 'home_team_league'],
                    suffixes=('', '_home_team'))
        
        away_teams = teams.copy()
        away_teams = away_teams.add_prefix('away_team_')
        result_df = result_df.merge(away_teams, 
                    how='left', 
                    left_on=['away_team_id', 'season_start', 'league'], 
                    right_on=['away_team_id', 'away_team_season_end', 'away_team_league'],
                    suffixes=('', '_away_team'))
        
        
        result_df = result_df.drop(columns=['home_team_league', 'home_team_season_start', 'home_team_season_end', 
                                            'away_team_league', 'away_team_season_start', 'away_team_season_end', 
                                            'home_team_country', 'away_team_country'])

        # result_df = self.ohe_features_encode({'home_team_country': self.config.categories['country'], 
        #                                       'away_team_country': self.config.categories['country']}, 
        #                                       result_df)
        
        
        # print(result_df['away_team_country_Monaco'].head())
        # print(result_df.select_dtypes(exclude=['number', 'category']).columns)

        # home_teams_columns_to_drop = set(home_teams.columns) - set(home_teams.select_dtypes(include=['number']).columns)
        # away_teams_columns_to_drop = set(away_teams.columns) - set(away_teams.select_dtypes(include=['number']).columns)

        # result_df = result_df.drop(columns=list(set(result_df.columns) - set(home_teams_columns_to_drop)))
        # result_df = result_df.drop(columns=list(set(result_df.columns) - set(away_teams_columns_to_drop)))


        # result_df = result_df.drop(columns=[col_name for col_name in result_df.columns if col_name.endswith('_away_team') or col_name.endswith('_home_team')])
        
        # # Вывод столбцов с пропусками и количество пропусков в процентах
        # columns_with_nan = result_df.columns[result_df.isnull().any()].tolist()
        # nan_percentage = result_df.isnull().mean() * 100

        # for column in columns_with_nan:
        #     print(f"Столбец '{column}' имеет {nan_percentage[column]:.2f}% пропусков")


        return result_df
    

    def add_date_features(self, matches) -> pd.DataFrame:

        matches['day'] = matches['match_date'].dt.day
        matches['month'] = matches['match_date'].dt.month
        matches['year'] = matches['match_date'].dt.year

        
        matches['day_of_week'] = matches['match_date'].dt.dayofweek
        matches['day_of_year'] = matches['match_date'].dt.dayofyear
        matches['quarter'] = matches['match_date'].dt.quarter

        
        matches['day_sin'] = np.sin(2 * np.pi * matches['day'] / 31)
        matches['day_cos'] = np.cos(2 * np.pi * matches['day'] / 31)
        matches['month_sin'] = np.sin(2 * np.pi * matches['month'] / 12)
        matches['month_cos'] = np.cos(2 * np.pi * matches['month'] / 12)

           
        return matches


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
        input = self._inputs[idx]

        if self.transforms is not None:
            input = self.transforms(input)

        if self.set_type.name != 'test':
            return {'input': input.astype(np.float32), 'target': self._targets[idx]}
        else:
            return {'input': input.astype(np.float32)}
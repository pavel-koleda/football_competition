import os

from easydict import EasyDict

from utils.enums import PreprocessingType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

data_cfg = EasyDict()

# Path to the directory with dataset files
data_cfg.path_to_data = os.path.join(ROOT_DIR, 'data', 'Football Dataset')

# Label mapping
data_cfg.label_mapping = {
    'draw': 0,
    'home': 1,
    'away': 2
}
data_cfg.classes_num = 3

# Categories

data_cfg.categories = {
    'league':   ['La Liga', 
                 'Serie A', 
                 'Premier League', 
                 'Bundesliga', 
                 'UEFA Europa League', 
                 'Ligue 1', 
                 'UEFA Europa Conference League', 
                 'UEFA Champions League'],
                 
    'position': ['Defender', 
                 'Midfielder', 
                 'Forward', 
                 'Goalkeeper'],

    'country':  ['Germany',
                 'France',
                 'Moldova',
                 'Romania',
                 'Serbia',
                 'Italy',
                 'Azerbaijan',
                 'Czech Republic',
                 'Portugal',
                 'Switzerland',
                 'Ukraine',
                 'Kazakhstan',
                 'England',
                 'Belgium',
                 'Spain',
                 'Austria',
                 'Greece',
                 'Turkey',
                 'Bulgaria',
                 'Poland',
                 'Netherlands',
                 'Croatia',
                 'Cyprus',
                 'Wales',
                 'Norway',
                 'Sweden',
                 'Russia',
                 'Israel',
                 'Slovenia',
                 'Denmark',
                 'Scotland',
                 'Hungary',
                ' Slovakia',
                 'Monaco',
                 'Belarus']
}

# Training configuration
data_cfg.preprocess_type = PreprocessingType.standardization
# data_cfg.preprocess_type = PreprocessingType.normalization

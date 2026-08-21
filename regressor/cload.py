import os

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsRegressor

import joblib


data_dir = '/work/cvcs2026/LZMM/COLET/'

gaze_df = pd.read_csv(os.path.join(data_dir, 'gaze_dataset.csv'), sep=',')
pupil_df = pd.read_csv(os.path.join(data_dir, 'pupil_dataset.csv'), sep=',')
blinks_df = pd.read_csv(os.path.join(data_dir, 'blinks_dataset.csv'), sep=',')
annotation_df = pd.read_csv(os.path.join(data_dir, 'annotation_dataset.csv'), sep=',')

transformer = ColumnTransformer(
    transformers=[

    ], 
    remainder='passthrough', 
    n_jobs=-1
)

knr = KNeighborsRegressor(
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ('preprocessing', transformer),
        ('regressor', knr)
    ]
)

# training (fit)

joblib.dump(
    pipeline, 
    os.path.join(os.path.dirname(__file__), 'weights/cl.pkl')
)
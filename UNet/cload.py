import os

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor

import joblib


gaze_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'COLET/gaze_dataset.csv'), sep=',')
pupil_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'COLET/pupil_dataset.csv'), sep=',')
blinks_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'COLET/blinks_dataset.csv'), sep=',')
annotation_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'COLET/annotation_dataset.csv'), sep=',')

transformer = ColumnTransformer(transformers=[

    ], 
    remainder='passthrough', 
    n_jobs=-1
)

rfc = RandomForestRegressor(
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessing', transformer),
    ('training', rfc)
])

# training (fit)

joblib.dump(
    pipeline, 
    os.path.join(os.path.dirname(__file__), 'regressor-weights/cl.pkl')
)
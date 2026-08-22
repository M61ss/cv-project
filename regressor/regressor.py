import os

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsRegressor

import joblib

from dataset import X_train, X_test, y_train, y_test


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

pipeline.fit(X_train, y_train)

joblib.dump(
    pipeline, 
    os.path.join(os.path.dirname(__file__), 'weights/cl.pkl')
)
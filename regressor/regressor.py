import os

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

import joblib

from dataset import X_train, X_test, y_train, y_test


def evaluate_regressor(model, x_test, y_test, binary=True):
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    if (y_test > -1).all() & (y_pred > -1).all():
        msle = mean_squared_log_error(y_test, y_pred)
    else:
        msle = "N/A (y_test contiene valori <= -1)"
    r2 = r2_score(y_test, y_pred)

    print("")
    print(f"MAE:    {mae:.4f}")
    print(f"MSE:    {mse:.4f}")
    print(f"MSLE:   {msle:.4f}")
    print(f"R2:     {r2:.4f}")
    print("")
    print("Remember: R2 score is in [-inf, 1]. R2<0 --> BAD MODEL.")

    return y_pred


parameters = {
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size' : [10, 20, 30, 40, 50],
    'n_neighbors' : [1, 3, 5, 10, 20, 30],
    'p' : [1, 2, 3, 4, 5],
    'weights' : ['uniform', 'distance']
}

gscv = GridSearchCV(KNeighborsRegressor(n_jobs=-1), parameters, cv=5, n_jobs=-1)

print('Starting training...')
gscv.fit(X_train, y_train)
evaluate_regressor(gscv, X_test, y_test, binary=False)

print('Saving model...')
joblib.dump(
    gscv, 
    os.path.join(os.path.dirname(__file__), 'weights/cload.pkl')
)
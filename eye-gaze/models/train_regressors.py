import os
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_regressors(X, y, save_dir='./outputs/models'):
    os.makedirs(save_dir, exist_ok=True)
    dt = DecisionTreeRegressor(random_state=0)
    rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    dt.fit(X, y)
    rf.fit(X, y)
    dt_path = os.path.join(save_dir, 'decision_tree_regressor.joblib')
    rf_path = os.path.join(save_dir, 'random_forest_regressor.joblib')
    joblib.dump(dt, dt_path)
    joblib.dump(rf, rf_path)
    return {'decision_tree': dt_path, 'random_forest': rf_path}

def evaluate_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'mse': float(mse), 'r2': float(r2)}

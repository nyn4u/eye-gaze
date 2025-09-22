"""Top-level pipeline script to run the whole flow.
Usage:
python scripts/run_pipeline.py --img_dir ./images --out_dir ./outputs
"""
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

from data_processing import build_dataset_from_images
from models.train_regressors import train_regressors, evaluate_regressor
from models.train_classifier import train_classifier
from cpd.cpd_algorithms import cpd_adaptive_cusum, cpd_rolling_stat
from utils.io_utils import save_csv

def inject_anomalies(ts, anomaly_rate=0.05, magnitude=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n = len(ts)
    k = max(1, int(n * anomaly_rate))
    idxs = sorted(rng.choice(n, k, replace=False).tolist())
    ts2 = ts.copy()
    for idx in idxs:
        dx = rng.uniform(-magnitude, magnitude)
        dy = rng.uniform(-magnitude, magnitude)
        ts2.at[idx, 'gaze_x'] = np.clip(ts2.at[idx, 'gaze_x'] + dx, 0.0, 1.0)
        ts2.at[idx, 'gaze_y'] = np.clip(ts2.at[idx, 'gaze_y'] + dy, 0.0, 1.0)
    return ts2, idxs

def evaluate_cpd(predicted, ground_truth, tolerance=2):
    gt_set = set(ground_truth)
    matched = set()
    tp = 0
    for p in predicted:
        for g in ground_truth:
            if abs(p - g) <= tolerance:
                tp += 1
                matched.add(g)
                break
    fp = max(0, len(predicted) - tp)
    fn = max(0, len(ground_truth) - len(matched))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (len(ground_truth) - fn) / len(ground_truth) if len(ground_truth) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}

def main(img_dir, out_dir, anomaly_rate):
    os.makedirs(out_dir, exist_ok=True)
    print('Building dataset...')
    df = build_dataset_from_images(img_dir)
    if df.empty:
        raise RuntimeError('No images found')
    ts = df.reset_index(drop=True)
    ts = ts[['gaze_x', 'gaze_y']]
    ts_injected, gt = inject_anomalies(ts, anomaly_rate=anomaly_rate)
    save_csv(ts_injected.assign(is_anomaly=0).assign(is_anomaly=[1 if i in gt else 0 for i in range(len(ts_injected))]), os.path.join(out_dir, 'gaze_ts.csv'))

    # Prepare features
    history = 3
    coords = ts_injected[['gaze_x', 'gaze_y']].values
    X, y = [], []
    for i in range(history, len(coords)):
        X.append(coords[i-history:i].flatten())
        y.append(coords[i])
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print('Training regressors...')
    model_paths = train_regressors(X_train, y_train, save_dir=os.path.join(out_dir, 'models'))
    dt = joblib.load(model_paths['decision_tree'])
    rf = joblib.load(model_paths['random_forest'])
    print('DecisionTree eval:', evaluate_regressor(dt, X_test, y_test))
    print('RandomForest eval:', evaluate_regressor(rf, X_test, y_test))

    # classifier
    y_labels = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        if i + history in gt:
            y_labels[i] = 1
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_labels, test_size=0.2, random_state=0)
    nn_path = train_classifier(Xc_train, yc_train, Xc_test, yc_test, save_path=os.path.join(out_dir, 'models', 'nn_classifier.h5'))
    print('Saved NN classifier to', nn_path)

    # CPD on gaze_x
    series = ts_injected['gaze_x'].values
    cusum_cps = cpd_adaptive_cusum(series, threshold=3.0, drift=0.02)
    roll_cps = cpd_rolling_stat(series, window=max(5, int(len(series)*0.05)), alpha=0.01)
    print('CUSUM CPS:', cusum_cps)
    print('ROLL CPS:', roll_cps)

    print('Evaluations:')
    print('CUSUM:', evaluate_cpd(cusum_cps, gt))
    print('ROLL :', evaluate_cpd(roll_cps, gt))
    save_csv(pd.DataFrame({'cusum': cusum_cps}), os.path.join(out_dir, 'cusum.csv'))
    save_csv(pd.DataFrame({'roll': roll_cps}), os.path.join(out_dir, 'roll.csv'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./images')
    parser.add_argument('--out_dir', type=str, default='./outputs')
    parser.add_argument('--anomaly_rate', type=float, default=0.05)
    args = parser.parse_args()
    main(args.img_dir, args.out_dir, args.anomaly_rate)

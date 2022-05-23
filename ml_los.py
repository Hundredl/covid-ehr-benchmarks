import math
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeRegressor

from app.utils import metrics

"""
Tasks:
- mortality outcome
- los

Models:
- logistic regression (sklearn) !TOFIX
- random forest (sklearn)
- xgboost (xgboost)
- catboost (catboost)
- gbdt (sklearn)
- autogluon (automl models)
"""

RANDOM_SEED = 42


def train(x, y, method):
    if method == "xgboost":
        model = xgb.XGBRegressor(verbosity=0, n_estimators=1000, learning_rate=0.1)
        model.fit(x, y, eval_metric="auc")
    elif method == "gbdt":
        method = GradientBoostingRegressor(random_state=RANDOM_SEED)
        model = method.fit(x, y)
    elif method == "random_forest":
        method = RandomForestRegressor(random_state=RANDOM_SEED, max_depth=2)
        model = method.fit(x, y)
    elif method == "decision_tree":
        model = DecisionTreeRegressor(random_state=RANDOM_SEED)
        model.fit(x, y)
    elif method == "catboost":
        model = CatBoostRegressor(
            iterations=2,
            learning_rate=1,
            depth=2,
            loss_function="RMSE",
            verbose=None,
            allow_writing_files=False,
        )
        model.fit(x, y)
    return model


def validate(x, y, model):
    y_pred = model.predict(x)
    # print(y_pred[0:10], y[0:10])
    evaluation_scores = metrics.print_metrics_regression(y, y_pred, verbose=0)
    return evaluation_scores


def test(x, y, model):
    y_pred = model.predict(x)
    evaluation_scores = metrics.print_metrics_regression(y, y_pred, verbose=0)
    return evaluation_scores


if __name__ == "__main__":
    data_path = "./dataset/tongji/processed_data/"

    x = pickle.load(open("./dataset/tongji/processed_data/x.pkl", "rb"))

    y = pickle.load(open("./dataset/tongji/processed_data/y.pkl", "rb"))

    x_lab_length = pickle.load(
        open("./dataset/tongji/processed_data/visits_length.pkl", "rb")
    )

    x = x.numpy()
    y = y.numpy()
    x_lab_length = x_lab_length.numpy()
    x_flat = []
    y_flat = []
    for i in range(len(x)):
        cur_visits = x_lab_length[i]
        for j in range(int(cur_visits)):
            x_flat.append(x[i][j])
            y_flat.append(y[i][j])
    x = np.array(x_flat)
    y = np.array(y_flat)
    y_outcome = y[:, 0]
    y_los = y[:, 1]

    num_folds = 10
    kfold_test = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED
    )
    method = "xgboost"
    all_history = {}
    test_performance = {"test_mad": [], "test_mse": [], "test_mape": []}
    for fold_test, (train_and_val_idx, test_idx) in enumerate(
        kfold_test.split(np.arange(len(x)), y_outcome)
    ):
        print("====== Test Fold {} ======".format(fold_test + 1))
        kfold_val = StratifiedKFold(
            n_splits=num_folds - 1, shuffle=True, random_state=RANDOM_SEED
        )
        all_history["test_fold_{}".format(fold_test + 1)] = {}
        for fold_val, (train_idx, val_idx) in enumerate(
            kfold_val.split(
                np.arange(len(train_and_val_idx)), y_outcome[train_and_val_idx]
            )
        ):
            history = {"val_mad": [], "val_mse": [], "val_mape": []}
            model = train(x[train_idx], y_los[train_idx], method)
            val_evaluation_scores = validate(x[val_idx], y_los[val_idx], model)
            history["val_mad"].append(val_evaluation_scores["mad"])
            history["val_mse"].append(val_evaluation_scores["mse"])
            history["val_mape"].append(val_evaluation_scores["mape"])

            # print("y!", y_outcome[test_idx])

            test_evaluation_scores = test(x[test_idx], y_los[test_idx], model)
            test_performance["test_mad"].append(test_evaluation_scores["mad"])
            test_performance["test_mse"].append(test_evaluation_scores["mse"])
            test_performance["test_mape"].append(test_evaluation_scores["mape"])
            print(
                f"Performance on test set {fold_test+1}: \
                MAD = {test_evaluation_scores['mad']}, \
                MSE = {test_evaluation_scores['mse']}, \
                MAPE = {test_evaluation_scores['mape']}"
            )
            # print(history)
            all_history["test_fold_{}".format(fold_test + 1)][
                "fold{}".format(fold_val + 1)
            ] = history

    # Calculate average performance on 10-fold test set
    # print(test_performance)
    test_mad_list = np.array(test_performance["test_mad"])
    test_mse_list = np.array(test_performance["test_mse"])
    test_mape_list = np.array(test_performance["test_mape"])

    # print(test_mad_list)

    print("====================== RESULT ======================")
    print(
        "MAD: mean={:.3f}, std={:.3f}".format(test_mad_list.mean(), test_mad_list.std())
    )
    print(
        "MSE: mean={:.3f}, std={:.3f}".format(test_mse_list.mean(), test_mse_list.std())
    )
    print(
        "MAPE: mean={:.3f}, std={:.3f}".format(
            test_mape_list.mean(), test_mape_list.std()
        )
    )
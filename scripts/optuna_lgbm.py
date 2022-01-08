#!/usr/bin/env python
# coding: utf-8

import argparse

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_model(trial):
    """
    Make LGBM regression model

    Args:
        trial: Trial object that handles parameter suggestions.

    Returns:

    """
    model = LGBMRegressor(
        num_leaves=trial.suggest_int("num_leaves", 10, 200),
        learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        n_estimators=10 ** trial.suggest_int("log_n_estimators", 1, 4),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        random_state=1996,
        n_jobs=15,
    )
    return model


def objective(trial):
    """
    Objective function for hyperparameter optimization of LightGBM using Optuna.

    Args:
        trial: Trial object that handles parameter suggestions

    Returns:
        Validation error.

    """
    df = pd.read_csv("everything.csv")
    df = df.drop("Unnamed: 0", axis=1)
    features = np.array(
        df[
            [
                "LATITUDE",
                "LONGITUDE",
                "DISCOVERY_DOY",
                "STAT_CAUSE_CODE",
                "temp",
                "dutr",
                "prcp",
                "snow",
            ]
        ]
    )
    labels = np.array(df["FIRE_SIZE"])

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.10, shuffle=True, random_state=1996
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, shuffle=True, random_state=1996
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)

    model = make_model(trial)
    model.fit(X_train, y_train)

    prediction = model.predict(X_val)
    score = mean_absolute_error(y_val, prediction)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="optuna.py",
        description="Do hyperparameter optimization on LightGBM model",
    )
    parser.add_argument(
        "-n", "--ntrials", help="Number of trials", required=True, type=int, default=200
    )
    values = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=values.ntrials, timeout=10 * 60 * 60)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("   {}: {}".format(key, value))

    joblib.dump(study, f"lgbm_study_trials_{values.ntrials}.pkl")

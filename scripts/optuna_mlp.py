#!/usr/bin/env python
# coding: utf-8

import argparse

import joblib
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from scripts.utils import make_model


def objective(trial):
    """
    Objective function for hyperparameter optimization of MLP using Optuna.

    Args:
        trial: Trial object that handles parameter suggestions.

    Returns:
        Score: Validation score of trained model.
    """
    tf.keras.backend.clear_session()
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

    X_train = tf.convert_to_tensor(X_train)
    X_test = tf.convert_to_tensor(X_test)
    X_val = tf.convert_to_tensor(X_val)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)
    y_val = tf.convert_to_tensor(y_val)

    normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))

    model = make_model(trial, normalizer)
    err = "mean_absolute_error"
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    opt = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss=err, optimizer=opt, metrics=[err])

    batch_size = 4096
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=1e-3,
        patience=3,
        mode="min",
        restore_best_weights=False,
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        verbose=False,
        epochs=200,
        batch_size=batch_size,
        shuffle=True,
        callbacks=es_callback,
    )
    score = model.evaluate(X_val, y_val, verbose=0)
    return score[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="optuna.py",
        description="Do hyperparameter optimization for MLP",
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

    joblib.dump(study, f"mlp_study_trials_{values.ntrials}.pkl")

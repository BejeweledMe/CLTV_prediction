import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import matplotlib.pyplot as plt
from typing import Iterable, Callable, Sized, Sequence, Union


def rmsle(y_true, y_pred):
    msle = mean_squared_log_error(y_true, y_pred)
    return np.sqrt(msle)


def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def eval_models(models: Union[Iterable, Sized],
                X: Iterable,
                y: Iterable,
                eval_metric: Callable,
                metric_name: str = '',
                model_name: str = '',
                verbose_each: bool = True) -> None:
    """
    Evaluate ensemble of models.
    Print result for each model if *verbose_each* and print mean result

    :param models: pretrained models
    :param X: data to predict (features)
    :param y: labels of data (to evaluate)
    :param eval_metric: function for evaluating
    :param metric_name: name of metric function to print
    :param model_name: name of model to print
    :param verbose_each: print result of each model, not only mean result
    :return: None
    """

    mean_score = 0.0

    for model in models:
        y_pred = model.predict(X)
        score = eval_metric(y, y_pred)
        mean_score += score / len(models)

        if verbose_each:
            print(f'Model {model_name} {metric_name} score : {score}')

    print(f'Models {model_name} mean {metric_name} score : {mean_score}')


def cv_sklearn_model(folds,
                     df: pd.DataFrame,
                     y: Sequence,
                     eval_metric: Callable,
                     model_class: Callable,
                     params=None,
                     metric_name: str = '',
                     model_name: str = '',
                     plot_importances: bool = False) -> list:
    """
    Train sklearn-supported models with cross-validation technique.
    Print CV validation score.
    Plot importance of features if *plot_importances*

    :param folds: sklearn folds suck as KFold to use folds.split()
    :param df: dataframe, X-data to split
    :param y: labels related to df
    :param eval_metric: function for evaluating
    :param model_class: class to create model
    :param params: dictionary of model parameters passed to model_class(**params)
    :param metric_name: name of metric function to print
    :param model_name: name of model to print
    :param plot_importances: plot importance of features or not
    :return: list of fitted models
    """

    if params is None:
        params = dict()
    cv_score = 0.0
    n_folds = folds.n_splits
    models = []
    importances = []

    for fold_, (tr_idx, val_idx) in enumerate(folds.split(df)):
        x_tr, y_tr = df.iloc[tr_idx].values, y[tr_idx]
        x_val, y_val = df.iloc[tr_idx].values, y[tr_idx]

        model = model_class(**params)
        model.fit(x_tr, y_tr)
        models.append(model)

        y_pred = model.predict(x_val)
        score = eval_metric(y_val, y_pred)
        cv_score += score / n_folds

        print(f'{model_name} {metric_name} score on fold {fold_} : {score}')

        if plot_importances:
            importances.append(model.feature_importances_)

    print(f'{model_name} CV {metric_name} score on {n_folds} folds : {cv_score}')

    if plot_importances:
        importances = np.mean(importances, axis=0)
        features = df.columns
        indices = np.argsort(importances)

        plt.figure(figsize=[16, 14])
        plt.title('Mean feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    return models


def seed_everything(seed=1337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

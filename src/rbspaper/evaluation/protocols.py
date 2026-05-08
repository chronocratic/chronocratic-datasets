__all__ = ['get_evaluation_protocol']

from collections.abc import Callable
import logging

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(name=__name__)


def get_evaluation_protocol(evaluation_protocol: str) -> Callable:
    """Returns the evaluation protocol function based on the provided protocol name."""
    protocols = {'svm': _fit_svc, 'ridge': _fit_ridge}
    try:
        return protocols[evaluation_protocol.lower()]
    except KeyError as exc:
        message = (
            f'Invalid evaluation protocol: {evaluation_protocol}. '
            f'Valid options are: {list(protocols.keys())}'
        )
        raise ValueError(message) from exc


def _fit_classifier(
    *,
    pipeline: Pipeline,
    param_grid: dict,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
) -> Pipeline:
    combined_features = np.concatenate([train_features, valid_features], axis=0)
    combined_labels = np.concatenate([train_labels, valid_labels], axis=0)

    test_fold = [-1] * len(train_features) + [0] * len(valid_features)

    predefined_split = PredefinedSplit(test_fold)

    grid_search = GridSearchCV(pipeline, param_grid, cv=predefined_split, n_jobs=-1)
    grid_search.fit(combined_features, combined_labels)

    best_classifier = grid_search.best_estimator_

    logger.info('Best parameters: %s', grid_search.best_params_)

    return best_classifier


def _fit_svc(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
) -> Pipeline:

    imputer = SimpleImputer(strategy='constant', fill_value=None)
    scaler = StandardScaler()
    classifier_object = SVC(
        kernel='rbf',
        coef0=0.0,
        gamma='scale',
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        max_iter=10_000_000,
        decision_function_shape='ovr',
        random_state=42,
        verbose=False,
    )

    pipeline = Pipeline([('imputer', imputer), ('scaler', scaler), ('svc', classifier_object)])

    param_grid = {'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

    return _fit_classifier(
        pipeline=pipeline,
        param_grid=param_grid,
        train_features=train_features,
        train_labels=train_labels,
        valid_features=valid_features,
        valid_labels=valid_labels,
    )


def _fit_ridge(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
) -> Pipeline:
    imputer = SimpleImputer(strategy='mean')

    train_features = np.asarray(imputer.fit_transform(train_features))
    valid_features = np.asarray(imputer.transform(valid_features))

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    valid_features = scaler.transform(valid_features)

    alpha_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    validation_scores = []
    for alpha in alpha_values:
        classifier_object = Ridge(random_state=42, alpha=alpha)
        classifier_object.fit(train_features, train_labels)
        valid_predictions = classifier_object.predict(valid_features)
        score = (
            np.sqrt(((valid_predictions - valid_labels) ** 2).mean())
            + np.abs(valid_predictions - valid_labels).mean()
        )
        validation_scores.append(score)

    best_alpha = alpha_values[np.argmin(validation_scores)]

    classifier_object = Ridge(random_state=42, alpha=best_alpha)

    pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('ridge', classifier_object),
        ]
    )

    combined_features = np.concatenate([train_features, valid_features], axis=0)
    combined_labels = np.concatenate([train_labels, valid_labels], axis=0)

    pipeline.fit(combined_features, combined_labels)

    return pipeline

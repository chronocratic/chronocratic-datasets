"""Representation analysis utilities for clean and attacked embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.rbspaper.evaluation.classification import classify_and_evaluate

TWO_DIMS = 2
MIN_CLASSES = 2


@dataclass(frozen=True)
class _VisualizationSample:
    features: np.ndarray
    labels: np.ndarray


def _ensure_2d(*, features: np.ndarray) -> np.ndarray:
    if features.ndim <= TWO_DIMS:
        return features
    return features.reshape(features.shape[0], -1)


def _ensure_1d_labels(*, labels: np.ndarray) -> np.ndarray:
    if labels.ndim == 1:
        return labels
    return labels.reshape(-1)


def _cosine_similarity(*, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    denominator = np.clip(left_norm * right_norm, a_min=1e-12, a_max=None)
    numerator = np.sum(left * right, axis=1, keepdims=True)
    return (numerator / denominator).reshape(-1)


def compute_shift_metrics(
    *, clean_features: np.ndarray, attacked_features: np.ndarray
) -> dict[str, float]:
    """Compute embedding drift metrics between clean and attacked representations."""
    clean_features = _ensure_2d(features=clean_features)
    attacked_features = _ensure_2d(features=attacked_features)

    if clean_features.shape != attacked_features.shape:
        message = (
            'Clean and attacked features must have identical shapes for shift analysis. '
            f'Got {clean_features.shape=} and {attacked_features.shape=}.'
        )
        raise ValueError(message)

    delta = attacked_features - clean_features
    l2 = np.linalg.norm(delta, axis=1)
    cosine_similarity = _cosine_similarity(left=clean_features, right=attacked_features)

    return {
        'mean_l2_shift': float(np.mean(l2)),
        'median_l2_shift': float(np.median(l2)),
        'max_l2_shift': float(np.max(l2)),
        'mean_cosine_similarity': float(np.mean(cosine_similarity)),
        'mean_cosine_shift': float(np.mean(1.0 - cosine_similarity)),
    }


def compute_geometry_metrics(*, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute class-conditional geometry statistics for representation quality."""
    features = _ensure_2d(features=features)
    labels = _ensure_1d_labels(labels=labels)

    unique_labels = np.unique(labels)
    if unique_labels.shape[0] < MIN_CLASSES:
        return {
            'num_classes': float(unique_labels.shape[0]),
            'mean_within_class_distance': 0.0,
            'mean_between_class_centroid_distance': 0.0,
            'centroid_margin': 0.0,
        }

    centroids: list[np.ndarray] = []
    within_distances: list[float] = []

    for label in unique_labels:
        class_features = features[labels == label]
        centroid = np.mean(class_features, axis=0)
        centroids.append(centroid)
        class_delta = class_features - centroid
        class_norm = np.linalg.norm(class_delta, axis=1)
        within_distances.extend(class_norm.tolist())

    centroids_matrix = np.stack(centroids, axis=0)
    pairwise_distances: list[float] = []
    for left_index in range(len(centroids)):
        for right_index in range(left_index + 1, len(centroids)):
            distance = np.linalg.norm(
                centroids_matrix[left_index] - centroids_matrix[right_index], ord=2
            )
            pairwise_distances.append(float(distance))

    mean_within = float(np.mean(within_distances)) if within_distances else 0.0
    mean_between = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0

    centroid_margin = 0.0
    if mean_within > 0:
        centroid_margin = mean_between / mean_within

    return {
        'num_classes': float(unique_labels.shape[0]),
        'mean_within_class_distance': mean_within,
        'mean_between_class_centroid_distance': mean_between,
        'centroid_margin': float(centroid_margin),
    }


def compute_linear_separability(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    evaluation_protocol: str = 'svm',
) -> dict[str, float]:
    """Run a linear probe on representation vectors."""
    return classify_and_evaluate(
        train_features=train_features,
        train_labels=train_labels,
        valid_features=valid_features,
        valid_labels=valid_labels,
        test_features=test_features,
        test_labels=test_labels,
        evaluation_protocol=evaluation_protocol,
    )


def _sample_for_visualization(
    *, features: np.ndarray, labels: np.ndarray, max_samples: int, seed: int
) -> _VisualizationSample:
    features = _ensure_2d(features=features)
    labels = _ensure_1d_labels(labels=labels)

    if features.shape[0] <= max_samples:
        return _VisualizationSample(features=features, labels=labels)

    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(features.shape[0], size=max_samples, replace=False)
    return _VisualizationSample(features=features[indices], labels=labels[indices])


def compute_low_dim_artifacts(
    *, features: np.ndarray, labels: np.ndarray, max_samples: int, seed: int
) -> dict[str, Any]:
    """Compute deterministic 2D projection artifacts for visual inspection."""
    sampled = _sample_for_visualization(
        features=features, labels=labels, max_samples=max_samples, seed=seed
    )

    pca = PCA(n_components=2, random_state=seed)
    pca_2d = pca.fit_transform(sampled.features)

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init='pca',
        learning_rate='auto',
        perplexity=min(30.0, max(5.0, sampled.features.shape[0] / 50.0)),
    )
    tsne_2d = tsne.fit_transform(sampled.features)

    return {
        'sample_count': int(sampled.features.shape[0]),
        'labels': sampled.labels.tolist(),
        'pca_2d': pca_2d.tolist(),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'tsne_2d': tsne_2d.tolist(),
    }

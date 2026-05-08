"""Encoding strategies for producing representations from trained models."""

__all__ = ['encode_data']

import torch

from src.rbspaper.evaluation.enums import TimeSeriesEvaluationDownstreamTaskEnum
from src.rbspaper.models.autotcl import AutoTCL
from src.rbspaper.models.cost import CoST
from src.rbspaper.models.ts2vec import TS2Vec

_ModelType = TS2Vec | AutoTCL | CoST


def encode_data(
    data: torch.Tensor, model: _ModelType, batch_size: int, num_workers: int, downstream_task: str
) -> torch.Tensor:
    """Encode input data using the provided model and encoding strategy.

    Args:
        data: Input tensor to encode.
        model: Trained model instance (TS2Vec, AutoTCL, or CoST).
        batch_size: Batch size for encoding.
        num_workers: Number of DataLoader workers.
        downstream_task: Downstream task identifier string.

    Returns:
        Encoded representations.

    Raises:
        NotImplementedError: If the model type is not supported.
    """
    if isinstance(model, TS2Vec):
        return _ts2vec_encode(
            model=model,
            data=data,
            batch_size=batch_size,
            downstream_task=downstream_task,
            num_workers=num_workers,
        )
    if isinstance(model, AutoTCL):
        return _auto_tcl_encode(
            model=model,
            data=data,
            batch_size=batch_size,
            downstream_task=downstream_task,
            num_workers=num_workers,
        )
    if isinstance(model, CoST):
        return _cost_encode(
            model=model,
            data=data,
            batch_size=batch_size,
            downstream_task=downstream_task,
            num_workers=num_workers,
        )
    message = f'Logic not implemented for this model instance type: {type(model)}'
    raise NotImplementedError(message)


# ----- Generic -----


def _generic_classification_encode(
    model: _ModelType, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """Encode using full-series pooling for classification."""
    return model.encode(
        data=data, batch_size=batch_size, num_workers=num_workers, encoding_window='full_series'
    )


def _generic_clustering_encode(
    model: _ModelType, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """Encode using full-series pooling for clustering."""
    return model.encode(
        data=data, batch_size=batch_size, num_workers=num_workers, encoding_window='full_series'
    )


def _generic_forecasting_encode(
    model: _ModelType, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """Encode using sliding windows for forecasting."""
    return model.encode(
        data=data,
        batch_size=batch_size,
        num_workers=num_workers,
        encoding_window=None,
        sliding_length=1,
        sliding_padding=200,
        causal=True,
    )


# ----- TS2Vec -----


def _ts2vec_classification_encode(
    model: TS2Vec, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """TS2Vec encoding for classification tasks."""
    return _generic_classification_encode(
        model=model, data=data, batch_size=batch_size, num_workers=num_workers
    )


def _ts2vec_clustering_encode(
    model: TS2Vec, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """TS2Vec encoding for clustering tasks."""
    return _generic_clustering_encode(
        model=model, data=data, batch_size=batch_size, num_workers=num_workers
    )


def _ts2vec_forecasting_encode(
    model: TS2Vec, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """TS2Vec encoding for forecasting tasks."""
    return _generic_forecasting_encode(
        model=model, data=data, batch_size=batch_size, num_workers=num_workers
    )


def _ts2vec_encode(
    model: TS2Vec, data: torch.Tensor, batch_size: int, num_workers: int, downstream_task: str
) -> torch.Tensor:
    """Dispatch TS2Vec encoding based on downstream task.

    Args:
        model: Trained TS2Vec model.
        data: Input data tensor.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        downstream_task: Task identifier string.

    Returns:
        Encoded representations.

    Raises:
        ValueError: If downstream_task is unknown.
    """
    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.CLASSIFICATION.value:
        return _ts2vec_classification_encode(
            model=model, data=data, batch_size=batch_size, num_workers=num_workers
        )
    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.CLUSTERING.value:
        return _ts2vec_clustering_encode(
            model=model, data=data, batch_size=batch_size, num_workers=num_workers
        )
    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value:
        return _ts2vec_forecasting_encode(
            model=model, data=data, batch_size=batch_size, num_workers=num_workers
        )
    message = f'Unknown downstream task: {downstream_task}'
    raise ValueError(message)


# ----- AutoTCL -----


def _auto_tcl_classification_encode(
    model: AutoTCL, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """AutoTCL encoding for classification tasks."""
    return _generic_classification_encode(
        model=model, data=data, batch_size=batch_size, num_workers=num_workers
    )


def _auto_tcl_clustering_encode(
    model: AutoTCL, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """AutoTCL encoding for clustering tasks."""
    return _generic_clustering_encode(
        model=model, data=data, batch_size=batch_size, num_workers=num_workers
    )


def _auto_tcl_forecasting_encode(
    model: AutoTCL, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """AutoTCL encoding for forecasting tasks."""
    return _generic_forecasting_encode(
        model=model, data=data, batch_size=batch_size, num_workers=num_workers
    )


def _auto_tcl_encode(
    model: AutoTCL, data: torch.Tensor, batch_size: int, num_workers: int, downstream_task: str
) -> torch.Tensor:
    """Dispatch AutoTCL encoding based on downstream task.

    Args:
        model: Trained AutoTCL model.
        data: Input data tensor.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        downstream_task: Task identifier string.

    Returns:
        Encoded representations.

    Raises:
        ValueError: If downstream_task is unknown.
    """
    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.CLASSIFICATION.value:
        return _auto_tcl_classification_encode(
            model=model, data=data, batch_size=batch_size, num_workers=num_workers
        )
    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.CLUSTERING.value:
        return _auto_tcl_clustering_encode(
            model=model, data=data, batch_size=batch_size, num_workers=num_workers
        )
    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value:
        return _auto_tcl_forecasting_encode(
            model=model, data=data, batch_size=batch_size, num_workers=num_workers
        )
    message = f'Unknown downstream task: {downstream_task}'
    raise ValueError(message)


# ----- CoST -----


def _cost_classification_encode(
    model: CoST, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """CoST encoding for classification tasks."""
    return _generic_classification_encode(
        model=model, data=data, batch_size=batch_size, num_workers=num_workers
    )


def _cost_clustering_encode(
    model: CoST, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """CoST encoding for clustering tasks."""
    return _generic_clustering_encode(
        model=model, data=data, batch_size=batch_size, num_workers=num_workers
    )


def _cost_forecasting_encode(
    model: CoST, data: torch.Tensor, batch_size: int, num_workers: int
) -> torch.Tensor:
    """CoST encoding for forecasting tasks."""
    return _generic_forecasting_encode(
        model=model, data=data, batch_size=batch_size, num_workers=num_workers
    )


def _cost_encode(
    model: CoST, data: torch.Tensor, batch_size: int, num_workers: int, downstream_task: str
) -> torch.Tensor:
    """Dispatch CoST encoding based on downstream task.

    Args:
        model: Trained CoST model.
        data: Input data tensor.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        downstream_task: Task identifier string.

    Returns:
        Encoded representations.

    Raises:
        ValueError: If downstream_task is unknown.
    """
    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.CLASSIFICATION.value:
        return _cost_classification_encode(
            model=model, data=data, batch_size=batch_size, num_workers=num_workers
        )
    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.CLUSTERING.value:
        return _cost_clustering_encode(
            model=model, data=data, batch_size=batch_size, num_workers=num_workers
        )
    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value:
        return _cost_forecasting_encode(
            model=model, data=data, batch_size=batch_size, num_workers=num_workers
        )
    message = f'Unknown downstream task: {downstream_task}'
    raise ValueError(message)

"""Factory for building pipeline models from parameter dataclasses."""

__all__ = ['build_model_from_parameters']

from dataclasses import asdict

import lightning.pytorch as pl

from src.rbspaper.configs.models import (
    AutoTCLModelParameters,
    CoSTModelParameters,
    ModelParameters,
    TS2VecModelParameters,
)
from src.rbspaper.models.autotcl import AutoTCL
from src.rbspaper.models.cost import CoST
from src.rbspaper.models.ts2vec import TS2Vec


def build_model_from_parameters(*, parameters: ModelParameters) -> pl.LightningModule:
    """Build a Lightning model from typed model parameters."""
    if isinstance(parameters, TS2VecModelParameters):
        return TS2Vec(**asdict(parameters))
    if isinstance(parameters, AutoTCLModelParameters):
        return AutoTCL(**asdict(parameters))
    if isinstance(parameters, CoSTModelParameters):
        return CoST(**asdict(parameters))

    message = f'Unsupported model parameters type: {type(parameters)}'
    raise ValueError(message)

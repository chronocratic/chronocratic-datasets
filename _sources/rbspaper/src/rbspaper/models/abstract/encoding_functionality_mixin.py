from __future__ import annotations

__all__ = ['EncodingFunctionalityMixin']

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812 # torch.nn.functional convention
from torch.utils.data import DataLoader, TensorDataset
import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.rbspaper.models.encoders.masking import MaskMode
from src.rbspaper.models.utils import (
    apply_slicing,
    concat_last_step_features,
    extract_features_from_batch,
    full_series_pooling,
    integer_pooling,
    multiscale_pooling,
    process_sliding_window,
)

_logger = logging.getLogger(__name__)
_EXPECTED_INPUT_DIMS = 3


class EncodingFunctionalityMixin:
    """Mixin that provides encoding functionality for time series models.

    Uses polymorphic strategy methods (_get_encoder, _get_eval_method, _get_slice)
    instead of string dispatch. Subclasses override these to provide model-specific
    behavior (CoST vs TS2Vec/AutoTCL).

    Attributes are intentionally declared here but implemented by subclasses:
    AutoTCL, TS2Vec, CoST.
    """

    device: torch.device  # torch.device from LightningModule
    _encoder: nn.Module
    _averaged_encoder: nn.Module

    def _get_encoder(self) -> nn.Module:
        """Return the encoder used for inference. Override for model-specific encoders."""
        return self._averaged_encoder

    def _get_eval_method(self) -> Callable[..., torch.Tensor]:
        """Return the evaluation function reference. Override for model-specific eval."""
        return self._evaluate_with_pooling

    def _get_slice(self, sliding_padding: int, sliding_length: int) -> slice | None:
        """Return the output slice for sliding windows. Override per model."""
        return slice(sliding_padding, sliding_padding + sliding_length)

    def _evaluate_with_pooling(
        self,
        input_tensor: torch.Tensor,
        mask: MaskMode | None = None,
        slicing: slice | None = None,
        encoding_window: str | int | None = None,
    ) -> torch.Tensor:
        """Evaluate the model with pooling.

        Args:
            input_tensor: Input tensor of shape (batch, seq_len, features).
            mask: Optional mask tensor.
            slicing: Optional slice to apply on the output.
            encoding_window: Specifies the pooling strategy.

        Returns:
            The output tensor after applying the specified pooling strategy.
        """
        output_tensor = self._encoder(
            x=input_tensor.to(self.device, non_blocking=True), mask_mode=mask
        )

        if encoding_window == 'full_series':
            output_tensor = full_series_pooling(tensor=output_tensor, slicing=slicing)
        elif encoding_window == 'multiscale':
            output_tensor = multiscale_pooling(tensor=output_tensor, slicing=slicing)
        elif isinstance(encoding_window, int):
            output_tensor = integer_pooling(
                tensor=output_tensor, encoding_window=encoding_window, slicing=slicing
            )
        else:
            output_tensor = apply_slicing(tensor=output_tensor, slicing=slicing)

        return output_tensor.cpu()

    def _evaluate_with_feature_concatenation(
        self,
        input_tensor: torch.Tensor,
        _mask: MaskMode | None = None,
        _slicing: slice | None = None,
        _encoding_window: str | int | None = None,
    ) -> torch.Tensor:
        output_trend_tensor, output_seasonality_tensor = self._encoder(
            x=input_tensor.to(self.device, non_blocking=True), mask_mode=None
        )
        output_tensor = concat_last_step_features(output_trend_tensor, output_seasonality_tensor)
        return output_tensor.cpu()

    def _compute_sliding_representations(
        self,
        input_tensor: torch.Tensor,
        sliding_length: int,
        sliding_padding: int,
        *,
        causal: bool,
        mask: MaskMode | None,
        encoding_window: str | int | None,
        num_samples: int,
        batch_size: int,
    ) -> torch.Tensor:
        all_representations = []
        calculation_buffer = []
        calculation_buffer_length = 0
        time_series_length = input_tensor.size(1)

        output_slice = self._get_slice(sliding_padding, sliding_length)
        eval_method = self._get_eval_method()

        # use tqdm to show progress bar
        for start_index in tqdm.tqdm(
            range(0, time_series_length, sliding_length),
            desc='Sliding inference',
            unit='window',
            leave=False,
        ):
            left_index = start_index - sliding_padding
            right_index = start_index + sliding_length + (sliding_padding if not causal else 0)
            sliding_window = process_sliding_window(
                input_tensor=input_tensor,
                left_index=left_index,
                right_index=right_index,
                time_series_length=time_series_length,
            )
            if num_samples < batch_size:
                if calculation_buffer_length + num_samples > batch_size:
                    concatenated_buffer = torch.cat(calculation_buffer, dim=0)
                    representations = eval_method(
                        input_tensor=concatenated_buffer,
                        mask=mask,
                        slicing=output_slice,
                        encoding_window=encoding_window,
                    )
                    all_representations += torch.split(representations, num_samples)
                    calculation_buffer = []
                    calculation_buffer_length = 0
                calculation_buffer.append(sliding_window)
                calculation_buffer_length += num_samples

            else:
                representations = eval_method(
                    input_tensor=sliding_window,
                    mask=mask,
                    slicing=output_slice,
                    encoding_window=encoding_window,
                )
                all_representations.append(representations)

        if num_samples < batch_size and calculation_buffer_length > 0:
            concatenated_buffer = torch.cat(calculation_buffer, dim=0)
            representations = eval_method(
                input_tensor=concatenated_buffer,
                mask=mask,
                slicing=output_slice,
                encoding_window=encoding_window,
            )
            all_representations += torch.split(representations, num_samples)

        concatenated_representations = torch.cat(all_representations, dim=1)

        if encoding_window == 'full_series':
            concatenated_representations = F.max_pool1d(
                concatenated_representations.transpose(1, 2).contiguous(),
                kernel_size=concatenated_representations.size(1),
            ).squeeze(1)
        return concatenated_representations

    def encode(
        self,
        data: torch.Tensor,
        batch_size: int,
        num_workers: int,
        mask: MaskMode | None = None,
        encoding_window: str | int | None = None,
        *,
        causal: bool = False,
        sliding_length: int | None = None,
        sliding_padding: int = 0,
    ) -> torch.Tensor:
        """Compute representations using the model.

        Args:
            data: Shape (n_instance, n_timestamps, n_features). Missing data set to NaN.
            batch_size: Batch size used for inference.
            num_workers: Number of workers used for data loading.
            mask: Mask for the encoder. One of 'binomial', 'continuous',
                'all_true', 'all_false', or 'mask_last'.
            encoding_window: Pooling strategy. 'full_series', 'multiscale',
                or an integer for the pooling kernel size.
            causal: If True, future information is not encoded.
            sliding_length: Sliding window length. If set, sliding inference
                is applied.
            sliding_padding: Contextual data length for each sliding window.

        Returns:
            The representations for data.
        """
        encoder = self._get_encoder()
        eval_method = self._get_eval_method()

        if encoder is None:
            msg = 'Model must be trained or loaded before encoding'
            raise RuntimeError(msg)
        if data.ndim != _EXPECTED_INPUT_DIMS:
            msg = 'Input data shape must be (n_instance, n_timestamps, n_features)'
            raise ValueError(msg)

        num_samples, _time_series_length, _ = data.shape

        original_training_state = encoder.training
        encoder.eval()

        dataset = TensorDataset(data)
        _logger.info(
            'building data loader with batch size %s and num workers %s', batch_size, num_workers
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        with torch.inference_mode():
            all_outputs = []
            for batch in tqdm.tqdm(
                loader, desc='Encoding data', unit='batch', leave=True, total=len(loader)
            ):
                input_tensor = extract_features_from_batch(batch)

                if sliding_length is not None:
                    representations = self._compute_sliding_representations(
                        input_tensor=input_tensor,
                        sliding_length=sliding_length,
                        sliding_padding=sliding_padding,
                        causal=causal,
                        mask=mask,
                        encoding_window=encoding_window,
                        num_samples=num_samples,
                        batch_size=batch_size,
                    )
                else:
                    representations = eval_method(
                        input_tensor=input_tensor, mask=mask, encoding_window=encoding_window
                    )

                    if encoding_window == 'full_series':
                        representations = representations.squeeze(1)

                all_outputs.append(representations.cpu())

            all_outputs = torch.cat(all_outputs, dim=0)

        encoder.train(original_training_state)
        return all_outputs

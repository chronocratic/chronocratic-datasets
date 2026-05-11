"""Abstract base configuration classes for dataset metadata.

Provides a frozen, validated hierarchy of Pydantic models for typed
dataset configuration. The base class defines common fields shared by
all dataset families; intermediate classes (ClassificationConfig,
ForecastingConfig) add category-specific fields; family-specific leaf
classes inherit from these intermediates.

The entire hierarchy is immutable (frozen=True) to prevent runtime
tampering with shared configuration instances. Use model_copy() for
creating runtime variants.
"""

from __future__ import annotations

import abc
import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from tscollection.datasets.enums.data import (
    DatasetFamily,
    SplitMode,
    SplittingStrategy,
)

__all__ = [
    'ArffFilePattern',
    'ClassificationConfig',
    'ClassificationFilePatterns',
    'DatasetConfig',
    'ForecastingConfig',
]

_SHA256_PATTERN = re.compile(r'^[0-9a-f]{64}$')


class ArffFilePattern(BaseModel):
    """Immutable ARFF file pattern for a single split.

    Attributes:
        arff: ARFF filename template with ``{dataset_name}`` placeholder.
            For example: ``'{dataset_name}_train.arff'``.
    """

    model_config = ConfigDict(frozen=True)

    arff: str


class ClassificationFilePatterns(BaseModel):
    """Immutable train/test file patterns for classification datasets.

    Attributes:
        train: File pattern for the training split.
        test: File pattern for the test split.
    """

    model_config = ConfigDict(frozen=True)

    train: ArffFilePattern
    test: ArffFilePattern


class DatasetConfig(BaseModel, abc.ABC):  # type: ignore[misc]
    """Abstract base configuration for all dataset families.

    Defines common fields shared by classification and forecasting
    configurations. This class cannot be instantiated directly; use
    concrete subclasses or intermediate classes (ClassificationConfig,
    ForecastingConfig) instead.

    Attributes:
        name: Human-readable dataset name (e.g., 'Coffee', 'ETTh1').
        family: Dataset family identifier (UCR, UEA, ETT, etc.).
        url: Download URL for the dataset archive or file.
        sha256: SHA256 checksum for integrity verification, or None if
            not yet available.
        num_classes: Number of distinct class labels. Required for
            classification datasets (>= 1), defaults to 0 for forecasting.
        data_form: Data representation format. Common values are
            'regular' (flat arrays) and 'nested' (multivariate).
        tasks: Tuple of supported task types for this dataset.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    family: DatasetFamily
    url: HttpUrl
    sha256: str | None = None
    num_classes: int = Field(default=0, ge=0)
    data_form: str | None = None
    tasks: tuple[str, ...]

    @field_validator('sha256')
    @classmethod
    def validate_sha256_format(cls, v: str | None) -> str | None:
        """Validate sha256 is a 64-character hexadecimal string or None."""
        if v is not None and not _SHA256_PATTERN.match(v):
            raise ValueError(
                'sha256 must be a 64-character lowercase hexadecimal string'
            )
        return v

    @field_validator('url')
    @classmethod
    def validate_https_only(cls, v: HttpUrl) -> HttpUrl:
        """Enforce HTTPS-only URLs to prevent MITM attacks on dataset downloads."""
        if str(v).startswith('http://'):
            raise ValueError(
                'URL must use HTTPS scheme (HTTP is not permitted for '
                'dataset downloads)'
            )
        return v

    @abc.abstractmethod
    def _config_validate(self) -> None:
        """Abstract validation hook that forces subclasses to be concrete.

        This method makes DatasetConfig truly abstract, preventing direct
        instantiation. Subclasses should call ``super()._config_validate()``
        if they add their own validation logic.
        """

    @property
    def cache_key(self) -> str:
        """Return a cache identifier derived from the download URL and checksum.

        Returns:
            A string in the format ``<url>:<sha256>`` or
            ``<url>:unknown`` when no checksum is available.
        """
        return f'{str(self.url)}:{self.sha256 or "unknown"}'


class ClassificationConfig(DatasetConfig):
    """Base configuration for classification datasets.

    Extends DatasetConfig with fields required by classification pipelines:
    target column name, file patterns, and splitting strategy.

    Attributes:
        target_col_name: Name of the ARFF column containing class labels.
        file_patterns: Frozen nested model defining train/test ARFF
            filename templates.
        split_strategy: How to construct the train/test split. Defaults
            to ``AS_DEFINED`` (use the archive's built-in splits).
    """

    target_col_name: str
    file_patterns: ClassificationFilePatterns
    split_strategy: SplittingStrategy = SplittingStrategy.AS_DEFINED

    def _config_validate(self) -> None:
        """Validate classification-specific constraints."""
        pass

    @model_validator(mode='after')
    def validate_classification_fields(self) -> ClassificationConfig:
        """Ensure data_form is set for classification configs.

        Classification datasets require a data_form value ('regular' or
        'nested') to determine how ARFF data is parsed. This check is
        on the intermediate class because it references fields owned by
        both DatasetConfig (data_form) and ClassificationConfig itself.

        Raises:
            ValueError: If data_form is None.
        """
        if self.data_form is None:
            raise ValueError(
                'Classification datasets must specify data_form'
                ' (e.g., "regular" or "nested")'
            )
        return self


class ForecastingConfig(DatasetConfig):
    """Base configuration for forecasting datasets.

    Extends DatasetConfig with fields required by forecasting pipelines:
    split mode and boundaries, default sequence length, and prediction
    horizon.

    Attributes:
        split_mode: How train/valid/test boundaries are expressed --
            ``INDEXED`` for absolute row positions, ``FRACTIONAL`` for
            proportional splits.
        split_bounds: Tuple of three values defining the boundaries.
            For INDEXED mode: ``(train_end, valid_end, test_end)`` as
            integers. For FRACTIONAL mode:
            ``(train_frac, valid_frac, test_frac)`` as floats summing to 1.0.
        default_seq_len: Default input window length for sliding-window
            datasets. Must be >= 1.
        default_horizon: Default prediction horizon. Must be >= 1.
    """

    split_mode: SplitMode
    split_bounds: tuple[int, ...] | tuple[float, ...]
    default_seq_len: int = Field(ge=1, default=128)
    default_horizon: int = Field(ge=1, default=96)

    def _config_validate(self) -> None:
        """Validate forecasting-specific constraints."""
        pass

    @field_validator('split_bounds', mode='before')
    @classmethod
    def validate_no_booleans_in_bounds(
        cls, v: tuple[int, ...] | tuple[float, ...]
    ) -> tuple[int, ...] | tuple[float, ...]:
        """Reject boolean values in split_bounds.

        In Python, bool is a subclass of int, so isinstance(True, int)
        returns True. This validator explicitly rejects booleans to prevent
        accidental coercion (e.g., split_bounds=(True, False, True)
        becoming (1, 0, 1)).

        Raises:
            ValueError: If any element is a boolean.
        """
        if any(isinstance(b, bool) for b in v):
            raise ValueError(
                'split_bounds must not contain boolean values '
                '(bool is a subclass of int in Python)'
            )
        return v

    @model_validator(mode='after')
    def validate_split_consistency(self) -> ForecastingConfig:
        """Validate split_bounds structure and consistency with split_mode.

        The length check is here (in a model_validator) rather than in a
        field_validator because field_validators do not run on default
        values in Pydantic v2. This ensures the constraint is enforced
        even when subclasses provide a default.

        Checks enforced:
        - split_bounds must have exactly 3 elements.
        - For FRACTIONAL mode: values must sum to ~1.0 and each >= 0.05.
        - For INDEXED mode: values must all be integers.

        Raises:
            ValueError: If the split bounds are invalid or inconsistent
                with the split mode.
        """
        if len(self.split_bounds) != 3:
            raise ValueError(
                f'split_bounds must have exactly 3 elements, '
                f'got {len(self.split_bounds)}'
            )
        if self.split_mode == SplitMode.FRACTIONAL:
            total = sum(self.split_bounds)  # type: ignore[arg-type]
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f'Fractional split_bounds must sum to 1.0, got {total}'
                )
            if any(b < 0.05 for b in self.split_bounds):  # type: ignore[union-attr]
                raise ValueError(
                    'Each fractional split component must be >= 0.05 '
                    f'(got {self.split_bounds})'
                )
        elif self.split_mode == SplitMode.INDEXED:
            if not all(isinstance(b, int) for b in self.split_bounds):
                raise ValueError(
                    'INDEXED split_bounds must be integers, got '
                    f'{[type(b).__name__ for b in self.split_bounds]}'
                )
        return self

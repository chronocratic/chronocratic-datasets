"""Tests for ETT forecasting wrapper.

Verifies that ETTDataset correctly inherits from
FlexibleTimeSeriesDatasetSingleFile, injects ForecastingStrategySingleFile,
and yields (input_tensor, target_tensor) tuples with correct shapes.
"""


def test_ett_yields_input_target(synthetic_forecast_data):
    """ETTDataset yields (input, target) pairs with correct shapes."""
    from chronocratic.datasets.datatypes.ett import ETTDataset

    ds = ETTDataset(data=synthetic_forecast_data, seq_len=96, step=1, forecast_horizon=24)
    inp, tgt = ds[0]

    # Check input shape: (seq_len, features) = (96, 7)
    assert inp.shape[0] == 96
    assert inp.shape[1] == 7

    # Check target shape: (forecast_horizon, features) = (24, 7)
    assert tgt.shape[0] == 24
    assert tgt.shape[1] == 7


def test_ett_length(synthetic_forecast_data):
    """ETTDataset length matches expected sequence count."""
    from chronocratic.datasets.datatypes.ett import ETTDataset

    # 200 timesteps, seq_len=96, forecast_horizon=24, step=1
    # Max start index: 200 - 96 - 24 = 80, so indices 0..80 => 81 sequences
    ds = ETTDataset(data=synthetic_forecast_data, seq_len=96, step=1, forecast_horizon=24)
    assert len(ds) > 0
    # The strategy counts sequences; verify it matches the corrected count
    assert len(ds) == 81


def test_ett_forecast_horizon(synthetic_forecast_data):
    """ETTDataset target has forecast_horizon timesteps."""
    from chronocratic.datasets.datatypes.ett import ETTDataset

    horizon = 48
    ds = ETTDataset(data=synthetic_forecast_data, seq_len=96, step=1, forecast_horizon=horizon)
    _, tgt = ds[0]
    assert tgt.shape[0] == horizon


def test_ett_windows_start_every_step():
    """Window k covers rows ``k * step`` onwards and its target follows it."""
    import numpy as np

    from chronocratic.datasets.datatypes.ett import ETTDataset

    data = np.arange(200, dtype=np.float32).reshape(-1, 1)
    ds = ETTDataset(data=data, seq_len=32, step=32, forecast_horizon=16)

    assert len(ds) == 5  # starts 0, 32, 64, 96, 128; 160 + 48 > 200
    for k in range(len(ds)):
        inp, tgt = ds[k]
        start = k * 32
        np.testing.assert_array_equal(inp.numpy()[:, 0], np.arange(start, start + 32))
        np.testing.assert_array_equal(tgt.numpy()[:, 0], np.arange(start + 32, start + 48))

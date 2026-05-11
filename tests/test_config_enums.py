"""Tests for DatasetFamily and SplitMode enums (CFG-03).

Verifies that the new StrEnum classes have the correct members,
serialize to strings, and are importable from the enums package.
"""

import pytest


def test_dataset_family_has_all_members() -> None:
    """CFG-03: DatasetFamily enum has all 8 rbspaper family values."""
    from tscollection.datasets.enums.data import DatasetFamily

    expected = {
        'UCR': 'ucr',
        'UEA': 'uea',
        'ETT': 'ett',
        'ELECTRICITY': 'electricity',
        'WEATHER': 'weather',
        'EXCHANGE': 'exchange',
        'TRAFFIC': 'traffic',
        'ILLNESS': 'illness',
    }
    for member_name, value in expected.items():
        assert hasattr(DatasetFamily, member_name)
        assert DatasetFamily[member_name].value == value


def test_dataset_family_member_count() -> None:
    """CFG-03: DatasetFamily has exactly 8 members."""
    from tscollection.datasets.enums.data import DatasetFamily

    assert len(DatasetFamily) == 8


def test_dataset_family_is_strenum() -> None:
    """CFG-03: DatasetFamily members are StrEnum instances."""
    from tscollection.datasets.enums.data import DatasetFamily

    assert isinstance(DatasetFamily.UCR, str)
    assert DatasetFamily.UCR == 'ucr'


def test_split_mode_has_members() -> None:
    """CFG-03: SplitMode enum has INDEXED and FRACTIONAL values."""
    from tscollection.datasets.enums.data import SplitMode

    assert SplitMode.INDEXED.value == 'indexed'
    assert SplitMode.FRACTIONAL.value == 'fractional'


def test_split_mode_member_count() -> None:
    """CFG-03: SplitMode has exactly 2 members."""
    from tscollection.datasets.enums.data import SplitMode

    assert len(SplitMode) == 2


def test_split_mode_is_strenum() -> None:
    """CFG-03: SplitMode members are StrEnum instances."""
    from tscollection.datasets.enums.data import SplitMode

    assert isinstance(SplitMode.INDEXED, str)
    assert SplitMode.INDEXED == 'indexed'


def test_import_from_enums_package() -> None:
    """CFG-03: DatasetFamily and SplitMode are importable from tscollection.datasets.enums."""
    from tscollection.datasets.enums import DatasetFamily, SplitMode

    assert DatasetFamily.UCR == 'ucr'
    assert SplitMode.INDEXED == 'indexed'

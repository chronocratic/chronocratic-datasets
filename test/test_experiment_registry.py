"""Tests for the restructured experiment registry."""

import warnings

import pytest

from experiment_instances.instances import (
    EXPERIMENTS_REGISTRY,
    ExperimentInstance,
    _EXPERIMENT_ID_ALIASES,
    get_experiment_instance,
    list_experiment_ids,
)
from src.rbspaper.attacks.enums import AttackFamily


class TestRegistryStructure:
    def test_registry_has_model_scoped_ids(self) -> None:
        ids = list_experiment_ids()
        assert 'ts2vec' in ids
        assert 'autotcl' in ids
        assert len(ids) == 2  # only model IDs, no model_attack combos

    def test_no_old_ids_in_registry(self) -> None:
        ids = list_experiment_ids()
        old_ids = [
            'ts2vec_fgsm',
            'ts2vec_pgd',
            'ts2vec_bim',
            'ts2vec_multi',
            'autotcl_fgsm',
            'autotcl_pgd',
            'autotcl_multi',
        ]
        for old_id in old_ids:
            assert old_id not in ids, f'Old ID {old_id!r} still in registry'

    def test_attack_families_structure(self) -> None:
        inst = EXPERIMENTS_REGISTRY['ts2vec']
        assert isinstance(inst.attack_families, dict)
        assert AttackFamily.WHITE_BOX in inst.attack_families
        assert isinstance(inst.attack_families[AttackFamily.WHITE_BOX], tuple)

    def test_attack_params_property_flattens(self) -> None:
        inst = EXPERIMENTS_REGISTRY['ts2vec']
        flat = inst.attack_params
        for family in inst.attack_families.values():
            for attack in family:
                assert attack in flat


class TestAliasResolution:
    def test_alias_emits_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            get_experiment_instance(experiment_id='ts2vec_fgsm')
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert 'deprecated' in str(w[0].message).lower()
            assert 'ts2vec' in str(w[0].message)

    def test_alias_resolves_correct_instance(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            aliased = get_experiment_instance(experiment_id='ts2vec_fgsm')
            direct = get_experiment_instance(experiment_id='ts2vec')
            # Should be the same registry object
            assert aliased is direct

    def test_all_aliases_map(self) -> None:
        assert _EXPERIMENT_ID_ALIASES['ts2vec_fgsm'] == 'ts2vec'
        assert _EXPERIMENT_ID_ALIASES['ts2vec_pgd'] == 'ts2vec'
        assert _EXPERIMENT_ID_ALIASES['ts2vec_bim'] == 'ts2vec'
        assert _EXPERIMENT_ID_ALIASES['ts2vec_multi'] == 'ts2vec'
        assert _EXPERIMENT_ID_ALIASES['autotcl_fgsm'] == 'autotcl'
        assert _EXPERIMENT_ID_ALIASES['autotcl_pgd'] == 'autotcl'
        assert _EXPERIMENT_ID_ALIASES['autotcl_multi'] == 'autotcl'


class TestFamilyFiltering:
    def test_no_filter_returns_instance(self) -> None:
        inst = get_experiment_instance(experiment_id='ts2vec')
        # Should return the registry object (not a copy)
        assert inst is EXPERIMENTS_REGISTRY['ts2vec']

    def test_white_box_filter_returns_copy(self) -> None:
        inst = get_experiment_instance(experiment_id='ts2vec', attack_family=AttackFamily.WHITE_BOX)
        # Should be a different object
        assert inst is not EXPERIMENTS_REGISTRY['ts2vec']
        # Should only have white_box family
        assert list(inst.attack_families.keys()) == [AttackFamily.WHITE_BOX]

    def test_filtered_instance_has_attacks(self) -> None:
        inst = get_experiment_instance(experiment_id='ts2vec', attack_family=AttackFamily.WHITE_BOX)
        attacks = inst.attack_params
        assert len(attacks) > 0
        # FGSM, PGD, BIM are white-box
        names = [a.name for a in attacks]
        assert 'fgsm' in names
        assert 'pgd' in names
        assert 'bim' in names

    def test_unknown_family_returns_empty(self) -> None:
        # If we query black_box for ts2vec (which has no black_box attacks)
        inst = get_experiment_instance(experiment_id='ts2vec', attack_family=AttackFamily.BLACK_BOX)
        assert inst.attack_params == ()


class TestUnknownExperiment:
    def test_unknown_id_raises(self) -> None:
        with pytest.raises(KeyError, match='Unknown experiment'):
            get_experiment_instance(experiment_id='nonexistent')

import dataclasses

import pytest

from problem_miner.config import DEFAULT_CONFIG, PipelineConfig


def test_default_config_has_sane_types():
    assert isinstance(DEFAULT_CONFIG.hdbscan_min_cluster_size, int)
    assert isinstance(DEFAULT_CONFIG.junk_avg_word_threshold, float)
    assert isinstance(DEFAULT_CONFIG.embedding_model, str)


def test_config_is_immutable():
    # frozen=True -- catches accidental mutation of shared config at
    # runtime (e.g. one caller changing a threshold for everyone else).
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONFIG.hdbscan_min_cluster_size = 999


def test_config_can_be_overridden_via_constructor_not_mutation():
    custom = PipelineConfig(hdbscan_min_cluster_size=5)
    assert custom.hdbscan_min_cluster_size == 5
    assert DEFAULT_CONFIG.hdbscan_min_cluster_size == 10  # unaffected

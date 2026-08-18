"""PSI drift calculation."""

import numpy as np
import pytest
from monitoring.drift import (
    classify,
    population_stability_index,
    psi_from_profile,
    quantile_bin_edges,
)


class TestPSI:

    def test_identical_distributions_give_near_zero(self):
        rng = np.random.default_rng(1)
        values = rng.normal(100, 15, 5000)
        assert population_stability_index(values, values) < 0.01

    def test_a_sample_from_the_same_distribution_is_stable(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(100, 15, 10000)
        current = rng.normal(100, 15, 1000)
        assert population_stability_index(reference, current) < 0.10

    def test_a_strong_shift_exceeds_the_alert_threshold(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(100, 15, 10000)
        current = rng.normal(160, 15, 1000)      # mean moved four std devs
        assert population_stability_index(reference, current) > 0.25

    def test_psi_is_never_negative(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(0, 1, 5000)
        for shift in (-3, -1, 0, 1, 3):
            current = rng.normal(shift, 1, 1000)
            assert population_stability_index(reference, current) >= 0

    def test_empty_bins_do_not_raise(self):
        reference = np.concatenate([np.zeros(500), np.ones(500)])
        current = np.zeros(200)
        value = population_stability_index(reference, current)
        assert np.isfinite(value)

    def test_empty_input_returns_zero(self):
        assert population_stability_index([], [1, 2, 3]) == 0.0
        assert population_stability_index([1, 2, 3], []) == 0.0

    def test_scale_of_the_sample_does_not_matter(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(0, 1, 10000)
        small = population_stability_index(reference, rng.normal(0, 1, 200))
        large = population_stability_index(reference, rng.normal(0, 1, 5000))
        assert abs(small - large) < 0.15, "PSI compares shares, not counts"


class TestBinEdges:

    def test_produces_the_requested_number_of_bins(self):
        rng = np.random.default_rng(1)
        edges = quantile_bin_edges(rng.normal(0, 1, 1000), bins=10)
        assert len(edges) == 11

    def test_outer_edges_are_open(self):
        rng = np.random.default_rng(1)
        edges = quantile_bin_edges(rng.normal(0, 1, 1000))
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_edges_are_strictly_increasing(self):
        rng = np.random.default_rng(1)
        edges = quantile_bin_edges(rng.exponential(1, 1000))
        assert all(a < b for a, b in zip(edges, edges[1:]))

    def test_binary_feature_collapses_safely(self):
        binary = np.array([0] * 800 + [1] * 200, dtype=float)
        edges = quantile_bin_edges(binary, bins=10)
        assert len(edges) >= 2
        assert all(a < b for a, b in zip(edges, edges[1:]))

    def test_constant_feature_collapses_to_one_bin(self):
        assert quantile_bin_edges(np.full(500, 7.0)) == [-np.inf, np.inf]


class TestClassify:

    @pytest.mark.parametrize('psi,expected', [
        (0.0, 'OK'), (0.05, 'OK'), (0.099, 'OK'),
        (0.10, 'WARN'), (0.20, 'WARN'), (0.249, 'WARN'),
        (0.25, 'ALERT'), (1.0, 'ALERT'),
    ])
    def test_thresholds(self, psi, expected):
        assert classify(psi) == expected

    def test_custom_thresholds(self):
        assert classify(0.15, warn=0.20, alert=0.40) == 'OK'


class TestProfileBasedPSI:

    @staticmethod
    def profile_from(values, bins=10):
        edges = quantile_bin_edges(values, bins=bins)
        counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
        return {
            'bin_edges': edges,
            'reference_share': (counts / counts.sum()).tolist(),
            'n_reference': int(counts.sum()),
        }

    def test_same_data_gives_near_zero(self):
        rng = np.random.default_rng(1)
        values = rng.normal(100, 15, 5000)
        assert psi_from_profile(self.profile_from(values), values) < 0.01

    def test_shifted_data_alerts(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(100, 15, 5000)
        assert psi_from_profile(self.profile_from(reference),
                                rng.normal(170, 15, 1000)) > 0.25

    def test_values_outside_the_training_range_are_counted(self):
        """Open outer edges mean out-of-range values are not silently dropped."""
        rng = np.random.default_rng(1)
        reference = rng.uniform(0, 1, 5000)
        profile = self.profile_from(reference)
        far_outside = np.full(500, 99.0)
        assert psi_from_profile(profile, far_outside) > 0.25
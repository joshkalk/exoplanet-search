import numpy as np
import pytest

from exoplanet_search.phase1c_diagnostics import (
    convergence_diagnostics,
    emcee_rank_bulk_ess,
    emcee_tail_ess,
    independent_ensemble_agreement,
    independent_ensemble_state_rhat,
    walker_health_diagnostics,
)
from exoplanet_search.phase1d_draws import _validate_authoritative_diagnostic_criteria
from exoplanet_search.phase1c_types import DIAGNOSTIC_METHODOLOGY_VERSION, PARAMETER_ORDER, Phase1CConfig


def test_ensemble_state_rhat_is_invariant_to_walker_permutation():
    chain = _iid_ensemble_chain(seed=1)
    rng = np.random.default_rng(2)
    permuted = chain.copy()
    for ensemble in range(permuted.shape[0]):
        permuted[ensemble] = permuted[ensemble, rng.permutation(permuted.shape[1])]

    original = independent_ensemble_state_rhat(chain)
    shuffled = independent_ensemble_state_rhat(permuted)

    assert original["worst_by_parameter"] == pytest.approx(shuffled["worst_by_parameter"], rel=0.0, abs=1.0e-12)
    assert original["maximum"] == pytest.approx(shuffled["maximum"], rel=0.0, abs=1.0e-12)


def test_ensemble_state_rhat_near_one_for_matched_and_fails_for_shifted_ensemble():
    matched = _iid_ensemble_chain(seed=3, scale=0.02)
    shifted = matched.copy()
    shifted[3, :, :, 0] += 1.0

    matched_report = independent_ensemble_state_rhat(matched)
    shifted_report = independent_ensemble_state_rhat(shifted)

    assert matched_report["maximum"] < 1.05
    assert shifted_report["worst_by_parameter"]["log_rp"] > 1.1


def test_independent_ensemble_agreement_flags_scale_and_tail_contamination():
    timing = _timing()
    config = Phase1CConfig(
        n_ensembles=4,
        n_walkers=8,
        convergence_ensemble_scale_ratio_max=2.0,
        convergence_tail_interval_overlap_minimum=0.6,
    )
    scale_mismatch = list(_iid_ensemble_chain(seed=4, scale=0.02))
    scale_mismatch[3][:, :, 0] = _base_vector()[0] + (scale_mismatch[3][:, :, 0] - _base_vector()[0]) * 8.0
    scale_agreement = independent_ensemble_agreement(scale_mismatch, timing, config, warmup_steps=0)

    contaminated = _iid_ensemble_chain(seed=5, scale=0.02)
    contaminated[3, :, -20:, 0] += 1.0
    tail_agreement = independent_ensemble_agreement(list(contaminated), timing, config, warmup_steps=0)

    assert scale_agreement["passed"] is False
    assert any(row["reason"] == "scale_mismatch" for row in scale_agreement["failures"])
    assert tail_agreement["passed"] is False
    assert any(row["reason"] in {"tail_interval_overlap", "scale_mismatch"} for row in tail_agreement["failures"])


def test_walker_health_flags_stuck_walker_but_not_ordinary_correlated_walkers():
    config = Phase1CConfig(n_ensembles=4, n_walkers=8)
    chain = _iid_ensemble_chain(seed=6, steps=80)
    log_prob = np.zeros(chain.shape[:3])
    acceptance = np.full((4, 8), 0.25)
    chain[3, 2, :, :] = 25.0
    log_prob[3, 2, :] = -1000.0
    acceptance[3, 2] = 0.0

    health = walker_health_diagnostics(chain, log_prob, acceptance, config, warmup_steps=10)
    assert health["severe_walker_count"] == 1
    assert health["severe_walkers"][0]["ensemble"] == 3
    assert health["severe_walkers"][0]["walker"] == 2

    correlated = np.cumsum(_iid_ensemble_chain(seed=7, steps=80, scale=0.01), axis=2)
    ordinary = walker_health_diagnostics(correlated, np.zeros(correlated.shape[:3]), acceptance, config, warmup_steps=10)
    assert ordinary["severe_walker_count"] == 0


def test_emcee_bulk_ess_decreases_for_more_autocorrelated_sequences():
    iid = _iid_ensemble_chain(seed=8, steps=160, scale=0.02)
    ar1 = _ar1_ensemble_chain(seed=8, steps=160, rho=0.95)

    iid_ess = emcee_rank_bulk_ess(iid)
    ar1_ess = emcee_rank_bulk_ess(ar1)

    assert iid_ess["all_available"] is True
    assert ar1_ess["all_available"] is True
    assert ar1_ess["minimum"] < iid_ess["minimum"]


def test_tail_ess_detects_persistent_tail_indicators_and_unavailable_tau():
    iid = _iid_ensemble_chain(seed=9, steps=160, scale=0.02)
    persistent = iid.copy()
    persistent[:, :, :80, 0] -= 5.0
    constant = np.zeros_like(iid)

    iid_tail = emcee_tail_ess(iid)
    persistent_tail = emcee_tail_ess(persistent)
    unavailable_tail = emcee_tail_ess(constant)

    assert iid_tail["all_available"] is True
    assert persistent_tail["all_available"] is True
    assert persistent_tail["combined_by_parameter"]["log_rp"] < iid_tail["combined_by_parameter"]["log_rp"]
    assert unavailable_tail["all_available"] is False
    assert unavailable_tail["combined_by_parameter"]["log_rp"] is None


def test_convergence_uses_new_policy_and_legacy_values_are_non_gating():
    config = Phase1CConfig(n_ensembles=4, n_walkers=8, convergence_ess_minimum=10.0, convergence_rhat_threshold=1.2)
    chain = _iid_ensemble_chain(seed=10, steps=120, scale=0.02).reshape((32, 120, len(PARAMETER_ORDER)))
    log_prob = np.zeros(chain.shape[:2])
    acceptance = np.full(32, 0.3)
    autocorr = _autocorr_report(config, tau=1.0, retained_steps=119)

    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        acceptance,
        autocorr,
        config,
        warmup_steps=1,
        posterior_stability={"passed": True},
        ensemble_agreement={"passed": True},
    )

    assert diagnostics["diagnostic_methodology_version"] == DIAGNOSTIC_METHODOLOGY_VERSION
    assert diagnostics["legacy_walker_as_chain_diagnostics"]["gating"] is False
    assert "no_severe_walker_pathology" in diagnostics["criteria"]


def test_old_diagnostic_policy_version_cannot_be_used_authoritatively():
    config = Phase1CConfig(run_id="primary")
    diagnostics = {
        "diagnostic_methodology_version": "old_walker_as_chain",
        "finite_log_probability_fraction": 1.0,
        "criteria": {
            "complete_valid_rhat": True,
            "complete_valid_bulk_ess": True,
            "complete_valid_tail_ess": True,
            "rhat_all_below_threshold": True,
            "ess_all_above_minimum": True,
            "tail_ess_all_above_minimum": True,
            "complete_valid_autocorrelation": True,
            "chain_length_exceeds_tau_multiple": True,
            "posterior_summary_stability": True,
            "independent_ensemble_agreement": True,
            "finite_log_probability_fraction_is_one": True,
            "no_severe_walker_pathology": True,
        },
    }
    history = {
        "rhat_max": 1.0,
        "bulk_ess_min": 1200.0,
        "tail_ess_min": 1200.0,
        "posterior_stability_passed": True,
        "independent_ensemble_agreement_passed": True,
        "complete_valid_autocorrelation": True,
        "chain_length_exceeds_tau_multiple": True,
        "no_severe_walker_pathology": True,
        "diagnostic_methodology_version": DIAGNOSTIC_METHODOLOGY_VERSION,
    }
    with pytest.raises(ValueError, match="current diagnostic methodology"):
        _validate_authoritative_diagnostic_criteria(config, diagnostics, history)


def _iid_ensemble_chain(*, seed: int, ensembles: int = 4, walkers: int = 8, steps: int = 120, scale: float = 0.02):
    rng = np.random.default_rng(seed)
    return _base_vector() + rng.normal(0.0, scale, size=(ensembles, walkers, steps, len(PARAMETER_ORDER)))


def _ar1_ensemble_chain(*, seed: int, ensembles: int = 4, walkers: int = 8, steps: int = 120, rho: float = 0.9):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.02, size=(ensembles, walkers, steps, len(PARAMETER_ORDER)))
    values = np.empty_like(noise)
    values[:, :, 0, :] = _base_vector() + noise[:, :, 0, :]
    for step in range(1, steps):
        values[:, :, step, :] = _base_vector() + rho * (values[:, :, step - 1, :] - _base_vector()) + noise[:, :, step, :]
    return values


def _base_vector():
    return np.asarray([-2.525, 2.14, 0.32, 0.30, 0.40, -9.43, 0.0, 0.0], dtype=float)


def _autocorr_report(config: Phase1CConfig, *, tau: float, retained_steps: int):
    rows = [
        {
            "ensemble": ensemble,
            "parameter": parameter,
            "tau": tau,
            "available": True,
            "retained_steps": retained_steps,
            "error": None,
        }
        for ensemble in range(config.n_ensembles)
        for parameter in PARAMETER_ORDER
    ]
    return {"rows": rows, "all_available": True, "worst_tau": tau, "unavailable_count": 0}


def _timing():
    from exoplanet_search.phase1c import synthetic_dataset

    _, timing, _ = synthetic_dataset(Phase1CConfig(n_ensembles=4, n_walkers=8))
    return timing

"""Distributed Cox regression, both backends.

The tolerance here is *statistical*, and stated in standard-error units
rather than absolute ones. Two optimizers with different
implementations do not stop at the same point even when asked the same
question, so an absolute bound would be a disguised assertion about
scipy's line search. What matters is whether the distributed encrypted
fit is the same estimate as the centralized one to any degree a
statistician would act on.
"""

from __future__ import annotations

import numpy as np
import pytest

from homomorphepy import have_backend, load_dlbcl, load_json, set_thread_env

set_thread_env(2)

pytest.importorskip("statsmodels", reason="install the 'stats' extra")

pytestmark = [
    pytest.mark.openfhe,
    pytest.mark.skipif(not have_backend(), reason="openfhe not installed"),
]

# The encrypted fit must sit well inside a hundredth of a standard
# error of the centralized one. Measured worst case is ~1.1e-03 SE.
SE_TOL = 1e-2


@pytest.fixture(scope="module")
def standard_errors():
    from statsmodels.duration.hazard_regression import PHReg

    from homomorphepy.examples.cox import COVARIATES

    df = load_dlbcl()
    fit = PHReg(
        df["time"].to_numpy(dtype=float),
        np.column_stack([df[c].to_numpy(dtype=float) for c in COVARIATES]),
        status=df["status"].to_numpy(dtype=int),
        strata=df["Subgroup"].cat.codes.to_numpy(),
        ties="efron",
    ).fit()
    return dict(zip(COVARIATES, np.asarray(fit.bse, dtype=float), strict=True))


@pytest.fixture(scope="module")
def threshold_fit():
    from homomorphepy.examples import cox

    return cox.run("threshold")


@pytest.fixture(scope="module")
def ckks_fit():
    from homomorphepy.examples import cox

    return cox.run("ckks")


class TestThresholdBackend:
    def test_converges(self, threshold_fit):
        assert threshold_fit.converged

    def test_matches_centralized_within_a_hundredth_of_an_se(
        self, threshold_fit, standard_errors
    ):
        for name, se in standard_errors.items():
            err = abs(
                threshold_fit.coefficients[name] - threshold_fit.centralized[name]
            )
            assert err / se < SE_TOL, f"{name}: {err:.2e} = {err / se:.1e} SE"

    def test_loglikelihood_matches_centralized(self, threshold_fit):
        # Flat near the optimum, so this agrees far more tightly than
        # the coefficients do.
        assert threshold_fit.loglik_encrypted == pytest.approx(
            threshold_fit.loglik_centralized, abs=1e-5
        )

    def test_no_party_holds_the_key(self, threshold_fit):
        m = threshold_fit.master
        assert len(m.sites) == 3
        assert not hasattr(m, "secret_share")
        assert all(s.secret_share is not None for s in m.sites)

    def test_sites_are_the_subgroups_in_protocol_order(self, threshold_fit):
        assert list(threshold_fit.site_sizes) == ["GCB", "ABC", "Type III"]
        assert threshold_fit.site_sizes == {"GCB": 115, "ABC": 71, "Type III": 49}
        assert threshold_fit.site_events == {"GCB": 54, "ABC": 49, "Type III": 30}

    def test_optimizer_actually_drove_the_protocol(self, threshold_fit):
        # Each call is a full encrypt / homomorphic-sum / threshold
        # decrypt round across three sites.
        assert threshold_fit.n_objective_calls > 20


class TestBackendsAgree:
    """Only the master class differs; the fit must not."""

    def test_same_estimates(self, ckks_fit, threshold_fit, standard_errors):
        for name, se in standard_errors.items():
            d = abs(ckks_fit.coefficients[name] - threshold_fit.coefficients[name])
            assert d / se < SE_TOL

    def test_same_objective(self, ckks_fit, threshold_fit):
        assert ckks_fit.objective_at_fit == pytest.approx(
            threshold_fit.objective_at_fit, abs=1e-5
        )

    def test_both_converge(self, ckks_fit, threshold_fit):
        assert ckks_fit.converged and threshold_fit.converged


class TestAgreesWithR:
    """Cross-language, against the exported R fit."""

    @pytest.fixture(scope="class")
    @staticmethod
    def r_fit():
        ref = load_json("cox_loglik").get("reference_fit")
        if ref is None:
            pytest.skip("fixture predates the reference_fit block")
        return ref

    def test_objective_matches_R(self, threshold_fit, r_fit):
        # The summed per-site negative log-likelihood at the optimum:
        # the same quantity, computed by two languages over two
        # implementations of the Cox partial likelihood.
        assert threshold_fit.objective_at_fit == pytest.approx(
            r_fit["objective"], abs=1e-4
        )

    def test_coefficients_match_R(self, threshold_fit, r_fit, standard_errors):
        for name, se in standard_errors.items():
            d = abs(threshold_fit.coefficients[name] - r_fit["coefficients"][name])
            assert d / se < SE_TOL, f"{name}: {d:.2e} = {d / se:.1e} SE"

    def test_R_and_python_centralized_fits_agree(self, threshold_fit, r_fit):
        # coxph vs PHReg on the identical cohort: no optimizer involved,
        # so this is the tightest cross-language comparison available.
        for name, value in r_fit["centralized"].items():
            assert threshold_fit.centralized[name] == pytest.approx(value, abs=1e-6)

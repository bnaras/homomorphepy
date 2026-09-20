"""Differential privacy layered on the encrypted protocols.

Two kinds of test here, deliberately separated.

The **accounting** is pure arithmetic -- zCDP composition, the
epsilon conversion, the finite-difference amplification factor -- and
runs in milliseconds with no crypto at all.

The **behavioral** claims need real fits through the encrypted
channel and are slow, so they are marked and deselected by default.
They assert an *ordering*, not values: DP noise is drawn afresh every
query, so two runs never agree on numbers even within one language,
and a value-level comparison against R would be meaningless.
"""

from __future__ import annotations

import math

import pytest

from homomorphepy import have_backend, set_thread_env

set_thread_env(2)

pytest.importorskip("statsmodels", reason="install the 'stats' extra")

from homomorphepy.examples import dp  # noqa: E402


class TestAccounting:
    """zCDP composition. No crypto, no fitting."""

    def test_amplification_factor_is_707_at_R_step(self):
        # sigma*sqrt(2)/(2h) at h = 1e-3, the figure the vignette quotes.
        assert dp.amplification_factor(1e-3) == pytest.approx(707.1, abs=0.5)

    def test_amplification_grows_as_the_step_shrinks(self):
        # Why scipy's default step (~1.5e-8) is catastrophic here: the
        # amplification is ~7e7 rather than ~7e2.
        assert dp.amplification_factor(1.49e-8) > 1e7
        assert dp.amplification_factor(1e-3) < 1e3

    def test_zcdp_conversion_matches_the_formula(self):
        rho, delta = 0.5, 1e-5
        expected = rho + 2 * math.sqrt(rho * math.log(1 / delta))
        assert dp.zcdp_to_epsilon(rho, delta) == pytest.approx(expected)

    def test_budget_composes_linearly_in_rho(self):
        one = dp.budget(1, sigma=1.0)
        ten = dp.budget(10, sigma=1.0)
        assert ten["rho_total"] == pytest.approx(10 * one["rho_total"])

    def test_smaller_sigma_costs_more_privacy(self):
        # rho = (Delta/sigma)^2 / 2: less noise, weaker guarantee.
        loud = dp.budget(100, sigma=1.0)["epsilon"]
        quiet = dp.budget(100, sigma=0.1)["epsilon"]
        assert quiet > loud

    def test_more_queries_cost_more_privacy(self):
        few = dp.budget(10, sigma=1.0)["epsilon"]
        many = dp.budget(1000, sigma=1.0)["epsilon"]
        assert many > few

    def test_zero_sigma_is_no_privacy(self):
        # sigma = 0 is the lossless protocol: exact release, infinite
        # epsilon. Reported rather than silently dividing by zero.
        assert dp.budget(10, sigma=0.0)["epsilon"] == math.inf

    def test_gradient_free_costs_budget_for_its_robustness(self):
        # The trade the vignette draws out: Nelder-Mead survives more
        # noise but needs far more queries, and queries are the thing
        # composition charges for.
        grad_based = dp.budget(150, sigma=1e-2)["epsilon"]
        grad_free = dp.budget(4000, sigma=1e-2)["epsilon"]
        assert grad_free > grad_based


@pytest.mark.slow
@pytest.mark.openfhe
@pytest.mark.skipif(not have_backend(), reason="openfhe not installed")
class TestBehavior:
    """Real fits through the encrypted channel. Minutes, not seconds."""

    def test_zero_sigma_reproduces_the_lossless_fit(self):
        # The mechanical check the R vignette leads with: with the
        # noise turned off, the DP protocol IS the lossless protocol.
        f = dp.fit_at_sigma(0.0, method="Nelder-Mead")
        assert f.max_abs_diff < 0.01
        assert f.sign_agreement == 5

    def test_gradient_based_search_collapses_before_gradient_free(self):
        # The ordering that is the point of the example. At sigma =
        # 1e-4 the gradient-free fit is still usable and the
        # gradient-based one has collapsed to the starting point.
        both = dp.compare_optimizers(1e-4)
        assert both["Nelder-Mead"].sign_agreement > both["BFGS"].sign_agreement
        assert both["Nelder-Mead"].max_abs_diff < both["BFGS"].max_abs_diff

    def test_gradient_free_pays_in_queries(self):
        both = dp.compare_optimizers(1e-4)
        assert both["Nelder-Mead"].n_queries > both["BFGS"].n_queries
        # ... and therefore in privacy budget, at equal sigma.
        assert both["Nelder-Mead"].epsilon > both["BFGS"].epsilon

    def test_accuracy_degrades_as_noise_grows(self):
        # Monotone in the qualitative sense: the quiet end recovers the
        # signs, the loud end does not.
        quiet = dp.fit_at_sigma(1e-4, method="Nelder-Mead")
        loud = dp.fit_at_sigma(1e-1, method="Nelder-Mead")
        assert quiet.sign_agreement >= loud.sign_agreement
        assert quiet.max_abs_diff <= loud.max_abs_diff

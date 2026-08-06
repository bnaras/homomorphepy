"""statsmodels PHReg vs survival::coxph — the Cox tier's foundation.

The threshold-Cox protocol aggregates each site's partial
log-likelihood evaluated at the current beta. In R that is
``coxph(init = beta, iter.max = 0)$loglik[1]``, called once per site
per optimizer evaluation. Porting the Cox examples therefore rests
entirely on Python being able to compute the same number.

A review asserted two things: that
``statsmodels PHReg(ties='efron').loglike(beta)`` is that number, and
that lifelines cannot produce it at all. Both are tested here rather
than taken on faith -- and the tie convention matters enormously,
since statsmodels defaults to Breslow while R defaults to Efron.

Needs no crypto backend: this is the statistical layer, checked
independently before any encryption is layered on top.
"""

from __future__ import annotations

import numpy as np
import pytest

from homomorphepy import load_dlbcl, load_json, site_order

statsmodels = pytest.importorskip(
    "statsmodels", reason="install the 'stats' extra for the Cox tier"
)
from statsmodels.duration.hazard_regression import PHReg  # noqa: E402


@pytest.fixture(scope="module")
def reference():
    return load_json("cox_loglik")


@pytest.fixture(scope="module")
def cohort(reference):
    df = load_dlbcl()
    covars = reference["covariates"]
    return {
        "time": df["time"].to_numpy(dtype=float),
        "status": df["status"].to_numpy(dtype=int),
        "X": np.column_stack([df[c].to_numpy(dtype=float) for c in covars]),
        "subgroup": df["Subgroup"].astype(str).to_numpy(),
    }


def loglike(c, mask, beta, ties="efron"):
    m = PHReg(c["time"][mask], c["X"][mask], status=c["status"][mask], ties=ties)
    return float(m.loglike(np.asarray(beta, dtype=float)))


class TestPHRegReproducesCoxph:
    def test_pooled_matches_R(self, reference, cohort):
        tol = reference["tolerance"]["value"]
        allrows = np.ones(len(cohort["time"]), dtype=bool)
        worst = max(
            abs(loglike(cohort, allrows, c["beta"]) - c["pooled"]["efron"])
            for c in reference["cases"]
        )
        assert worst < tol, f"max deviation {worst:.3e} exceeds {tol:.0e}"

    def test_per_site_matches_R(self, reference, cohort):
        tol = reference["tolerance"]["value"]
        worst = 0.0
        for case in reference["cases"]:
            for s in case["by_site"]:
                mask = cohort["subgroup"] == s["site"]
                assert mask.sum() == s["n"]
                worst = max(
                    worst, abs(loglike(cohort, mask, case["beta"]) - s["efron"])
                )
        assert worst < tol, f"max deviation {worst:.3e} exceeds {tol:.0e}"

    def test_summed_negative_loglik_matches_R(self, reference, cohort):
        # The quantity the protocol actually aggregates, so the one that
        # has to agree end to end -- not merely the pieces.
        tol = reference["tolerance"]["value"]
        for case in reference["cases"]:
            summed = -sum(
                loglike(cohort, cohort["subgroup"] == s["site"], case["beta"])
                for s in case["by_site"]
            )
            assert summed == pytest.approx(case["summed_nll_efron"], abs=tol)

    def test_site_order_is_the_protocol_order(self, reference):
        assert reference["site_order"] == site_order()


class TestTieConventionIsLoadBearing:
    """Efron vs Breslow is not a rounding-level choice on this data."""

    def test_breslow_default_would_be_badly_wrong(self, reference, cohort):
        # statsmodels defaults to ties='breslow'; R defaults to Efron.
        # Forgetting the argument is silent -- it returns a number, just
        # the wrong one. Quantify how wrong so nobody trims the option.
        allrows = np.ones(len(cohort["time"]), dtype=bool)
        worst = max(
            abs(
                loglike(cohort, allrows, c["beta"], ties="breslow")
                - c["pooled"]["efron"]
            )
            for c in reference["cases"]
        )
        assert worst > 1.0, (
            "Breslow and Efron now agree closely on this cohort; the "
            "ties='efron' argument may no longer be load-bearing"
        )

    def test_breslow_matches_R_breslow(self, reference, cohort):
        # Both conventions are exported, so we can confirm the
        # difference is the tie handling and not some other divergence.
        tol = reference["tolerance"]["value"]
        allrows = np.ones(len(cohort["time"]), dtype=bool)
        worst = max(
            abs(
                loglike(cohort, allrows, c["beta"], ties="breslow")
                - c["pooled"]["breslow"]
            )
            for c in reference["cases"]
        )
        assert worst < tol


class TestLifelinesCannotDoThis:
    """Why the mapping names statsmodels and not lifelines."""

    def test_no_loglik_at_arbitrary_beta(self):
        lifelines = pytest.importorskip("lifelines")

        cph = lifelines.CoxPHFitter()
        # log_likelihood_ is a post-fit attribute, not a function of
        # beta: there is no supported way to evaluate the partial
        # log-likelihood at a caller-supplied coefficient vector.
        assert not callable(getattr(cph, "log_likelihood_", None))

    def test_no_ties_option(self):
        import inspect

        lifelines = pytest.importorskip("lifelines")

        params = inspect.signature(lifelines.CoxPHFitter.__init__).parameters
        # Not even selectable, so matching R's default is not a matter
        # of passing the right argument.
        assert "ties" not in params

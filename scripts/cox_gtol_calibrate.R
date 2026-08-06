## Back-calculate a scipy-equivalent gtol from R's reltol = 1e-7.
##
## R's optim(method="BFGS") stops on a FUNCTION-VALUE criterion
## (reltol); scipy's BFGS stops on a GRADIENT-NORM criterion (gtol).
## They are not the same kind of test, so there is no exact algebraic
## conversion. But we can measure where R's criterion actually stops --
## and then set gtol so "converged" means the same thing empirically.
##
## Uses the cleartext summed-per-site objective. Established separately
## that encryption contributes ~1e-13 here, far below the ~1e-5 scale
## the finite-difference gradient operates at, so calibrating in the
## clear is valid and much faster.

suppressPackageStartupMessages({
    library(survival); library(homomorpheR); library(stats4)
})
data(DLBCL, package = "homomorpheR")

COVARS <- c("GCB_sig", "LN_sig", "Prolif_sig", "BMP6", "MHC2_sig")
FORM <- Surv(time, status) ~ GCB_sig + LN_sig + Prolif_sig + BMP6 + MHC2_sig
cph_control <- replace(coxph.control(), "iter.max", 0)
sites <- split(DLBCL, DLBCL$Subgroup)[levels(DLBCL$Subgroup)]

local_cox_nll <- function(data, beta) {
    fit <- tryCatch(coxph(FORM, data = data, init = beta, control = cph_control),
                    error = function(e) NULL)
    if (is.null(fit)) NA_real_ else -fit$loglik[1]
}
objective <- function(beta) sum(vapply(sites, local_cox_nll, numeric(1), beta = beta))

NDEPS <- 1e-3   # optim's default, and what the vignettes rely on

## Central-difference gradient at optim's own step size -- the same
## quantity scipy's gtol is applied to.
fd_grad <- function(beta, h = NDEPS) {
    vapply(seq_along(beta), function(j) {
        bp <- beta; bm <- beta
        bp[j] <- bp[j] + h; bm[j] <- bm[j] - h
        (objective(bp) - objective(bm)) / (2 * h)
    }, numeric(1))
}

cat("=== R optim(method='BFGS', reltol=1e-7), as the vignettes run it ===\n")
fit <- optim(par = rep(0, 5), fn = objective, method = "BFGS",
             control = list(reltol = 1e-7, ndeps = rep(NDEPS, 5)))
g <- fd_grad(fit$par)
cat(sprintf("  convergence code : %d  (%s)\n", fit$convergence,
            if (fit$convergence == 0) "converged" else "NOT converged"))
cat(sprintf("  fn evaluations   : %d\n", fit$counts[["function"]]))
cat(sprintf("  objective        : %.9f\n", fit$value))
cat(sprintf("  |grad|_2 at stop : %.6e\n", sqrt(sum(g^2))))
cat(sprintf("  |grad|_inf       : %.6e\n", max(abs(g))))
cat(sprintf("  coefficients     : %s\n",
            paste(sprintf("%.8f", fit$par), collapse = " ")))

## Reference: the centralized stratified fit the protocol reproduces.
agg <- coxph(Surv(time, status) ~ GCB_sig + LN_sig + Prolif_sig + BMP6 +
                 MHC2_sig + strata(Subgroup), data = DLBCL)
cat(sprintf("\n  max |optim - coxph| : %.3e\n",
            max(abs(fit$par - coef(agg)[COVARS]))))

## What gradient norm is even ATTAINABLE at this step size? The
## central-difference truncation error is O(h^2 * f'''), so there is a
## floor below which |grad| cannot be driven regardless of tolerance.
cat("\n=== attainable gradient floor at the optimum ===\n")
g_at_mle <- fd_grad(coef(agg)[COVARS])
cat(sprintf("  |grad|_2 at the exact coxph MLE : %.6e\n", sqrt(sum(g_at_mle^2))))
cat("  (a gtol below this cannot be met at ndeps = 1e-3, encrypted or not)\n")

cat("\n=== suggested scipy setting ===\n")
floor_norm <- sqrt(sum(g_at_mle^2))
stop_norm  <- sqrt(sum(g^2))
cat(sprintf("  R stops at |grad| = %.2e; floor is %.2e\n", stop_norm, floor_norm))
cat(sprintf("  gtol = %.0e keeps 'converged' comparable across languages\n",
            10^ceiling(log10(max(stop_norm, floor_norm) * 3))))

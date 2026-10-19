// r-script.js: the R cross-check script. Given a description of an analysis made in the app (no patient data:
// file name, outcome, predictors and their coding, missing-data methods, validation design, the numbers of the
// Testing-set rows, and the app's results), returns an R script that rebuilds the analysis independently from the
// user's CSV file with standard R packages (stats, survival, pROC, rms, riskRegression, prodlim, mice) and prints
// the app's and R's results side by side. Pure functions only (tested from Node and in the browser).

const rStr = s => JSON.stringify(String(s));
const rNum = v => (typeof v === 'number' && isFinite(v) ? String(v) : 'NA');
const rVec = (arr, f) => `c(${(arr || []).map(f).join(', ')})`;

function rList(obj) {
    const parts = Object.entries(obj).map(([k, v]) => {
        if (v === null || v === undefined) return `${k} = NULL`;
        if (Array.isArray(v)) return `${k} = ${v.length && typeof v[0] === 'number' ? rVec(v, rNum) : rVec(v, rStr)}`;
        if (typeof v === 'number') return `${k} = ${rNum(v)}`;
        if (typeof v === 'boolean') return `${k} = ${v ? 'TRUE' : 'FALSE'}`;
        if (typeof v === 'object') return `${k} = ${rList(v)}`;
        return `${k} = ${rStr(v)}`;
    });
    return `list(${parts.join(', ')})`;
}

// The analysis itself (independent of the app's code)
const R_CORE = String.raw`
# ------------------------------------------------------------------------------------------- set-up
need <- function(p) if (!requireNamespace(p, quietly = TRUE)) stop(sprintf('The R package "%s" is needed: install.packages("%s")', p, p), call. = FALSE)
for (p in c("survival", "pROC", "rms")) need(p)
if (S$outcome_type == "survival") { need("riskRegression"); need("prodlim"); suppressPackageStartupMessages({ library(prodlim); library(riskRegression) }) }  # Hist() must be found by name
if (S$missing == "mice") need("mice")
suppressPackageStartupMessages({ library(survival) })
set.seed(S$seed)
# Folder of this script (Rscript, or source() in RStudio), where the data file is looked for first
here <- local({ a <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd()) })
locate <- function(f) { for (p in c(f, file.path(here, f), file.path(getwd(), f))) if (file.exists(p)) return(p); stop(sprintf('Data file "%s" not found next to this script or in %s', f, getwd()), call. = FALSE) }

# Values the app treats as missing
MISSING <- c("", "na", "nan", "null", "none", "?", ".", "n/a")
read_data <- function(f) {
  d <- utils::read.csv(locate(f), check.names = FALSE, colClasses = "character", na.strings = character(0), fileEncoding = "UTF-8-BOM", strip.white = TRUE)
  for (j in names(d)) { v <- trimws(d[[j]]); v[tolower(v) %in% MISSING] <- NA; d[[j]] <- v }
  d
}
num <- function(v) suppressWarnings(as.numeric(v))

# ------------------------------------------------------------------------------------------- data
d <- read_data(S$data_file)
if (!is.na(S$n_rows) && nrow(d) != S$n_rows) stop(sprintf("The data file has %d rows; the analysis in the app had %d. Use the same file.", nrow(d), S$n_rows), call. = FALSE)
d$.row <- seq_len(nrow(d))
for (p in S$predictors) if (!is.null(p$derived_from)) d[[p$name]] <- ifelse(is.na(d[[p$derived_from]]), "Yes", "No")
pred_names <- vapply(S$predictors, function(p) p$name, "")
miss_cols <- setdiff(pred_names, names(d))
if (length(miss_cols)) stop(paste("Columns not found in the data file:", paste(miss_cols, collapse = ", ")), call. = FALSE)

# Outcome: 1/0 for a yes/no outcome; a number; or follow-up time with status 1 = event, 2 = competing event, 0 = censored
if (S$outcome_type == "survival") {
  d$.time <- num(d[[S$time_col]])
  ev <- d[[S$event_col]]
  d$.status <- ifelse(is.na(ev), NA, ifelse(ev == S$event_value, 1L, ifelse(!is.null(S$competing_value) & ev %in% S$competing_value, 2L, 0L)))
  usable <- !is.na(d$.time) & d$.time > 0 & !is.na(d$.status)
} else if (S$outcome_type == "binary") {
  d$.y <- ifelse(is.na(d[[S$outcome]]), NA, as.integer(d[[S$outcome]] == S$positive))
  usable <- !is.na(d$.y)
} else {
  d$.y <- num(d[[S$outcome]])
  usable <- !is.na(d$.y)
}
d <- d[usable, , drop = FALSE]
if (S$missing == "complete") d <- d[stats::complete.cases(d[, pred_names, drop = FALSE]), , drop = FALSE]

# Training and evaluation patients, as in the app
if (S$validation == "A") {
  test <- d[d$.row %in% S$test_rows, , drop = FALSE]; train <- d[!(d$.row %in% S$test_rows), , drop = FALSE]
} else if (S$validation == "B") {
  ext <- read_data(S$external_file)
  for (p in S$predictors) if (!is.null(p$derived_from)) ext[[p$name]] <- ifelse(is.na(ext[[p$derived_from]]), "Yes", "No")
  if (S$outcome_type == "survival") {
    ext$.time <- num(ext[[S$time_col]]); e2 <- ext[[S$event_col]]
    ext$.status <- ifelse(is.na(e2), NA, ifelse(e2 == S$event_value, 1L, ifelse(!is.null(S$competing_value) & e2 %in% S$competing_value, 2L, 0L)))
    ext <- ext[!is.na(ext$.time) & ext$.time > 0 & !is.na(ext$.status), , drop = FALSE]
  } else if (S$outcome_type == "binary") {
    ext$.y <- ifelse(is.na(ext[[S$outcome]]), NA, as.integer(ext[[S$outcome]] == S$positive)); ext <- ext[!is.na(ext$.y), , drop = FALSE]
  } else { ext$.y <- num(ext[[S$outcome]]); ext <- ext[!is.na(ext$.y), , drop = FALSE] }
  if (S$missing == "complete") ext <- ext[stats::complete.cases(ext[, pred_names, drop = FALSE]), , drop = FALSE]
  train <- d; test <- ext
} else { train <- d; test <- d }
if (S$validation == "A" && !is.null(S$test_check)) {
  ev_test <- if (S$outcome_type == "survival") sum(test$.status == 1) else if (S$outcome_type == "binary") sum(test$.y) else NA
  if (nrow(test) != S$test_check$n || (!is.na(ev_test) && !is.null(S$test_check$events) && ev_test != S$test_check$events))
    warning(sprintf("The Testing set rebuilt here has %d rows (%s events); the app reported %s rows (%s events). Check that this is the same data file.", nrow(test), ev_test, S$test_check$n, S$test_check$events), call. = FALSE)
}

# ------------------------------------------------------------------------------------------- missing values
# Single imputation exactly as the app does it: values fitted on the training patients, applied to everyone
fit_fill <- function(tr) {
  lapply(setNames(S$predictors, pred_names), function(p) {
    v <- tr[[p$name]]; v <- v[!is.na(v)]
    if (p$impute == "drop") return(list(drop = TRUE))
    strategy <- p$impute
    nums <- num(v)
    if (strategy %in% c("mean", "median", "knn") && any(is.na(nums))) strategy <- "mode"
    if (strategy == "knn") strategy <- if (p$type == "nominal") "mode" else "median"   # see the note printed below
    fill <- switch(strategy,
      mean = if (length(nums)) mean(nums) else 0,
      median = if (!length(nums)) 0 else if (p$type == "ordinal") sort(nums)[floor((length(nums) - 1) / 2) + 1] else stats::median(nums),
      mode = { if (!length(v)) (if (p$type == "nominal") "Unknown" else 0) else { tb <- table(factor(v, levels = unique(v))); m <- names(tb)[which.max(tb)]; if (p$type != "nominal" && !is.na(num(m))) num(m) else m } },
      constant = if (p$type == "nominal") (if (is.null(p$constant) || p$constant == "") "Missing" else p$constant) else (if (is.null(p$constant) || p$constant == "") 0 else num(p$constant)),
      stop("Unknown missing-data method ", strategy))
    list(drop = FALSE, fill = fill)
  })
}
apply_fill <- function(dd, plan) {
  for (p in S$predictors) {
    pl <- plan[[p$name]]
    if (isTRUE(pl$drop)) { dd <- dd[!is.na(dd[[p$name]]), , drop = FALSE]; next }
    dd[[p$name]][is.na(dd[[p$name]])] <- as.character(pl$fill)
  }
  dd
}
knn_used <- any(vapply(S$predictors, function(p) identical(p$impute, "knn"), TRUE))

# ------------------------------------------------------------------------------------------- coding of predictors
# Model frame for R's own models: numbers as numbers, ordinal predictors as their rank, categories as factors
# with the most frequent training category as the reference (as in the app's Cox model)
frame <- function(dd, ref) {
  out <- data.frame(row.names = seq_len(nrow(dd)))
  for (p in S$predictors) {
    v <- dd[[p$name]]
    if (p$type == "continuous") out[[p$name]] <- num(v)
    else if (p$type == "ordinal") out[[p$name]] <- match(v, p$levels) - 1
    else out[[p$name]] <- stats::relevel(factor(v, levels = ref[[p$name]]$levels), ref = ref[[p$name]]$reference)
  }
  out
}
frame_ref <- function(tr) lapply(setNames(S$predictors, pred_names), function(p) {
  if (p$type != "nominal") return(NULL)
  lv <- sort(unique(tr[[p$name]]))
  tb <- table(factor(tr[[p$name]], levels = lv)); list(levels = lv, reference = names(tb)[which.max(tb)])
})
# The app's own encoding (for its penalised logistic regression): standardised numbers and ranks (training mean and
# population SD), and one 0/1 column for every category
app_matrix <- function(dd, tr) {
  cols <- list()
  for (p in S$predictors) {
    if (p$type == "nominal") {
      for (lv in sort(unique(tr[[p$name]]))) cols[[paste0(p$name, "=", lv)]] <- as.numeric(dd[[p$name]] == lv)
    } else {
      x <- if (p$type == "ordinal") match(dd[[p$name]], p$levels) - 1 else num(dd[[p$name]])
      xt <- if (p$type == "ordinal") match(tr[[p$name]], p$levels) - 1 else num(tr[[p$name]])
      m <- mean(xt, na.rm = TRUE); s <- sqrt(mean((xt - m)^2, na.rm = TRUE)); if (!(s > 0)) s <- 1
      cols[[p$name]] <- round((x - m) / s, 5)
    }
  }
  as.matrix(as.data.frame(cols, check.names = FALSE))
}

# ------------------------------------------------------------------------------------------- reporting
RES <- data.frame(measure = character(), app = character(), r = character(), r_label = character(), stringsAsFactors = FALSE)
fmt <- function(v, d = 3) if (is.null(v) || length(v) == 0 || all(is.na(v))) "-" else if (length(v) == 2) sprintf(paste0("%.", d, "f to %.", d, "f"), v[1], v[2]) else sprintf(paste0("%.", d, "f"), v)
add <- function(measure, app, r, label = "R") RES[nrow(RES) + 1, ] <<- list(measure, fmt(app), fmt(r), label)
count <- function(measure, app, r) RES[nrow(RES) + 1, ] <<- list(measure, if (is.null(app)) "-" else as.character(app), as.character(r), "R")
calib <- function(y, p) {
  p <- pmin(pmax(p, 1e-12), 1 - 1e-12); lp <- stats::qlogis(p)
  list(slope = unname(stats::coef(stats::glm(y ~ lp, family = binomial))[2]), intercept = unname(stats::coef(stats::glm(y ~ offset(lp), family = binomial))[1]))
}
notes <- character()
if (knn_used) notes <- c(notes, "k-NN filling of missing values is replaced here by the median (numbers) or most frequent value (categories), so small differences are expected.")

# ===================================================================================== yes/no outcome
if (S$outcome_type == "binary") {
  need("pROC")
  prep_sets <- function(tr, te) {
    if (S$missing == "single") { plan <- fit_fill(tr); tr <- apply_fill(tr, plan); te <- apply_fill(te, plan) }
    list(tr = tr, te = te)
  }
  if (S$missing == "mice") {
    notes <- c(notes, sprintf("Multiple imputation: mice (predictive mean matching, %d imputations) fitted on the training patients only; predictions averaged over the imputations. MICE is random, so expect small differences from the app.", S$mice_m))
    both <- rbind(train, test); both$.is_test <- c(rep(FALSE, nrow(train)), rep(TRUE, nrow(test)))
    X <- both[, pred_names, drop = FALSE]
    for (p in S$predictors) X[[p$name]] <- if (p$type == "nominal") factor(X[[p$name]]) else if (p$type == "ordinal") match(X[[p$name]], p$levels) - 1 else num(X[[p$name]])
    imp <- mice::mice(X, m = S$mice_m, method = "pmm", ignore = both$.is_test, printFlag = FALSE, seed = S$seed)
    sets <- lapply(seq_len(S$mice_m), function(k) { cd <- mice::complete(imp, k); for (p in S$predictors) if (p$type == "ordinal") cd[[p$name]] <- as.character(p$levels[cd[[p$name]] + 1]) else cd[[p$name]] <- as.character(cd[[p$name]]); list(tr = cbind(cd[!both$.is_test, , drop = FALSE], .y = train$.y), te = cbind(cd[both$.is_test, , drop = FALSE], .y = test$.y)) })
  } else sets <- list(prep_sets(train, test))
  # Model 1: the app's model (logistic regression with the app's L2 penalty, on the app's encoding)
  fit_pen <- function(tr) {
    X <- app_matrix(tr, tr); y <- tr$.y; n <- nrow(X); lam <- S$lambda / 4   # two-class softmax with L2 = logistic with L2/4
    f <- function(b) { eta <- b[1] + X %*% b[-1]; mean(log1p(exp(-abs(eta))) + pmax(eta, 0) - y * eta) + lam * sum(b[-1]^2) }
    g <- function(b) { eta <- as.vector(b[1] + X %*% b[-1]); r <- stats::plogis(eta) - y; c(mean(r), as.vector(crossprod(X, r)) / n + 2 * lam * b[-1]) }
    o <- stats::optim(rep(0, ncol(X) + 1), f, g, method = "BFGS", control = list(maxit = 2000, reltol = 1e-14))
    list(b = o$par, tr = tr)
  }
  pred_pen <- function(m, te) as.vector(stats::plogis(m$b[1] + app_matrix(te, m$tr) %*% m$b[-1]))
  # Model 2: standard logistic regression (maximum likelihood, glm)
  fit_glm <- function(tr) { ref <- frame_ref(tr); fr <- frame(tr, ref); fr$.y <- tr$.y; list(fit = suppressWarnings(stats::glm(.y ~ ., data = fr, family = binomial)), ref = ref) }
  pred_glm <- function(m, te) as.vector(suppressWarnings(stats::predict(m$fit, newdata = frame(te, m$ref), type = "response")))
  p_pen <- rowMeans(sapply(sets, function(s) pred_pen(fit_pen(s$tr), s$te)))
  p_glm <- rowMeans(sapply(sets, function(s) pred_glm(fit_glm(s$tr), s$te)))
  y_te <- sets[[1]]$te$.y
  roc_pen <- pROC::roc(y_te, p_pen, levels = c(0, 1), direction = "<", quiet = TRUE)
  roc_glm <- pROC::roc(y_te, p_glm, levels = c(0, 1), direction = "<", quiet = TRUE)
  cal_pen <- calib(y_te, p_pen); cal_glm <- calib(y_te, p_glm)
  what <- if (S$validation == "C") "apparent (all patients)" else if (S$validation == "B") "validation cohort" else "Testing set"
  count(sprintf("Patients evaluated (%s)", what), APP$n, nrow(sets[[1]]$te))
  count("Events evaluated", APP$events, sum(y_te))
  add("ROC-AUC", APP$auc, as.numeric(pROC::auc(roc_pen)), "R, same model as the app")
  add("ROC-AUC 95% CI (app: bootstrap; R: DeLong)", APP$auc_ci, as.numeric(pROC::ci.auc(roc_pen, method = "delong"))[c(1, 3)], "R pROC (DeLong), same model")
  add("ROC-AUC", NULL, as.numeric(pROC::auc(roc_glm)), "R, standard glm")
  add("Brier score", APP$brier, mean((p_pen - y_te)^2), "R, same model as the app")
  add("Brier score", NULL, mean((p_glm - y_te)^2), "R, standard glm")
  add("Calibration slope", APP$slope, cal_pen$slope, "R, same model as the app")
  add("Calibration slope", NULL, cal_glm$slope, "R, standard glm")
  add("Calibration intercept", APP$intercept, cal_pen$intercept, "R, same model as the app")
  add("Calibration intercept", NULL, cal_glm$intercept, "R, standard glm")
  if (S$validation == "C") {
    # Harrell's bootstrap optimism correction with rms::validate (standard logistic regression)
    need("rms")
    tr <- sets[[1]]$tr; ref <- frame_ref(tr); fr <- frame(tr, ref); fr$.y <- tr$.y
    fit <- rms::lrm(.y ~ ., data = fr, x = TRUE, y = TRUE)
    v <- rms::validate(fit, B = S$B)
    add("Optimism-corrected C (AUC)", APP$auc_corrected, 0.5 + v["Dxy", "index.corrected"] / 2, sprintf("R rms::validate, %d resamples (standard model)", S$B))
    add("Optimism-corrected calibration slope", APP$slope_corrected, v["Slope", "index.corrected"], "R rms::validate (standard model)")
    if (S$missing == "single" && any(vapply(S$predictors, function(p) p$impute != "drop", TRUE))) notes <- c(notes, "rms::validate does not repeat the missing-value filling inside each resample (the app does), so its correction can be slightly smaller.")
    notes <- c(notes, "rms::validate corrects R's unpenalised model; the app corrects its own lightly penalised model, whose predictions are slightly less extreme, so its corrected calibration slope is usually a little closer to 1.")
  }
}

# ===================================================================================== numerical outcome
if (S$outcome_type == "continuous") {
  if (S$missing == "single") { plan <- fit_fill(train); tr <- apply_fill(train, plan); te <- apply_fill(test, plan) } else { tr <- train; te <- test }
  ref <- frame_ref(tr); fr <- frame(tr, ref); fr$.y <- tr$.y
  fit <- stats::lm(.y ~ ., data = fr)
  pr <- as.vector(stats::predict(fit, newdata = frame(te, ref)))
  # The app's model: ridge regression on the app's encoding, (1/2n) sum of squared errors + (lambda/2) |w|^2,
  # intercept unpenalised (solved exactly)
  Xa <- app_matrix(tr, tr); mx <- colMeans(Xa); my <- mean(tr$.y); Xc <- sweep(Xa, 2, mx)
  w <- solve(crossprod(Xc) / nrow(Xc) + diag(S$lambda, ncol(Xc)), crossprod(Xc, tr$.y - my) / nrow(Xc))
  pr_app <- as.vector(my + sweep(app_matrix(te, tr), 2, mx) %*% w)
  y <- te$.y
  r2 <- function(p) 1 - sum((y - p)^2) / sum((y - mean(y))^2)
  count("Patients evaluated", APP$n, length(y))
  add("R-squared", APP$r2, r2(pr_app), "R, same model as the app (ridge)")
  add("R-squared", NULL, r2(pr), "R, standard linear regression (lm)")
  add("RMSE", APP$rmse, sqrt(mean((y - pr_app)^2)), "R, same model as the app")
  add("RMSE", NULL, sqrt(mean((y - pr)^2)), "R, lm")
  add("MAE", APP$mae, mean(abs(y - pr_app)), "R, same model as the app")
  add("MAE", NULL, mean(abs(y - pr)), "R, lm")
  if (length(y) < 30) notes <- c(notes, sprintf("Only %d patients were evaluated: R-squared and errors are very imprecise, and an unpenalised lm with many predictors can differ greatly from the penalised model.", length(y)))
}

# ===================================================================================== time to event
if (S$outcome_type == "survival") {
  if (S$missing == "mice") {
    notes <- c(notes, "This check uses single imputation (median / most frequent value) instead of MICE, so expect small differences.")
    S$predictors <- lapply(S$predictors, function(p) { p$impute <- if (p$type == "nominal") "mode" else "median"; p })
    S$missing <- "single"
  }
  if (S$missing == "single") { plan <- fit_fill(train); tr <- apply_fill(train, plan); te <- apply_fill(test, plan) } else { tr <- train; te <- test }
  ref <- frame_ref(tr); ftr <- frame(tr, ref); fte <- frame(te, ref)
  h <- S$horizon
  competing <- !is.null(S$competing_value)
  ftr$.time <- tr$.time; ftr$.status <- tr$.status; fte$.time <- te$.time; fte$.status <- te$.status
  count("Patients evaluated", APP$n, nrow(te))
  count("Events evaluated", APP$events, sum(te$.status == 1))
  if (!competing) {
    fit <- survival::coxph(survival::Surv(.time, .status == 1) ~ . - .status - .time, data = ftr, ties = "efron")
    lp <- as.vector(stats::predict(fit, newdata = fte, type = "lp"))
    cc <- survival::concordance(survival::Surv(.time, .status == 1) ~ lp, data = cbind(fte, lp = lp), reverse = TRUE)
    add("Harrell's C-index", APP$c_index, cc$concordance, "R survival::concordance")
    risk <- 1 - as.vector(summary(survival::survfit(fit, newdata = fte), times = h, extend = TRUE)$surv)
    sc <- riskRegression::Score(list(Cox = risk), formula = Hist(.time, .status) ~ 1, data = fte, times = h, metrics = c("auc", "brier"), cens.model = "km", null.model = FALSE)
    add(sprintf("Time-dependent AUC at %s", h), APP$td_auc, sc$AUC$score$AUC[1], "R riskRegression::Score")
    add(sprintf("Brier score at %s", h), APP$brier, sc$Brier$score$Brier[sc$Brier$score$model == "Cox"][1], "R riskRegression::Score")
    km <- survival::survfit(survival::Surv(.time, .status == 1) ~ 1, data = fte)
    add(sprintf("Observed risk by %s (Kaplan-Meier)", h), APP$observed, 1 - summary(km, times = h, extend = TRUE)$surv, "R survival::survfit")
    add(sprintf("Mean predicted risk by %s", h), APP$predicted, mean(risk), "R")
    if (S$validation == "C") {
      need("rms")
      dd <- rms::datadist(ftr); options(datadist = dd)
      cf <- rms::cph(stats::as.formula(paste("survival::Surv(.time, .status == 1) ~", paste(sprintf("\x60%s\x60", pred_names), collapse = " + "))), data = ftr, x = TRUE, y = TRUE, surv = TRUE)
      v <- rms::validate(cf, B = S$B)
      add("Optimism-corrected C-index", APP$c_corrected, 0.5 + abs(v["Dxy", "index.corrected"]) / 2, sprintf("R rms::validate, %d resamples", S$B))
      add("Optimism-corrected calibration slope", APP$slope_corrected, v["Slope", "index.corrected"], "R rms::validate")
    }
    hr <- exp(stats::coef(fit))
    notes <- c(notes, "Hazard ratios from R (per unit or category vs the most frequent category):", paste0("  ", names(hr), ": ", sprintf("%.3f", hr)))
    if (S$validation == "A") notes <- c(notes, "The app weights the time-dependent AUC and Brier score with the training patients' censoring pattern; riskRegression uses the Testing set's, so small differences are expected.")
  } else {
    rhs <- paste(sprintf("\x60%s\x60", pred_names), collapse = " + ")
    fit <- riskRegression::CSC(stats::as.formula(paste("Hist(.time, .status) ~", rhs)), data = ftr, cause = 1)
    risk <- as.vector(riskRegression::predictRisk(fit, newdata = fte, times = h, cause = 1))
    sc <- riskRegression::Score(list(CSC = risk), formula = Hist(.time, .status) ~ 1, data = fte, times = h, cause = 1, metrics = c("auc", "brier"), cens.model = "km", null.model = FALSE)
    add(sprintf("Competing-risk AUC at %s", h), APP$td_auc, sc$AUC$score$AUC[1], "R riskRegression::Score")
    add(sprintf("Competing-risk Brier score at %s", h), APP$brier, sc$Brier$score$Brier[sc$Brier$score$model == "CSC"][1], "R riskRegression::Score")
    aj <- prodlim::prodlim(Hist(.time, .status) ~ 1, data = fte)
    add(sprintf("Observed risk by %s (Aalen-Johansen)", h), APP$observed, as.numeric(stats::predict(aj, times = h, cause = 1)), "R prodlim")
    add(sprintf("Mean predicted risk by %s", h), APP$predicted, mean(risk), "R riskRegression (cause-specific Cox)")
    # Wolbers' concordance, from its published definition (no standard R function)
    wolbers <- function(t, s, r) { cn <- 0; cp <- 0; for (i in which(s == 1)) { j <- which((t > t[i]) | (t == t[i] & s == 0) | (s == 2 & t <= t[i])); j <- j[j != i]; cp <- cp + length(j); cn <- cn + sum(r[i] > r[j]) + 0.5 * sum(r[i] == r[j]) }; cn / cp }
    add("Wolbers' C-index", APP$c_index, wolbers(te$.time, te$.status, risk), "R (published definition)")
    if (S$validation == "C") notes <- c(notes, "Optimism correction for competing-risk models has no standard R function: check the apparent values above instead.")
    if (S$validation == "A") notes <- c(notes, "The app weights the AUC and Brier score with the training patients' censoring pattern; riskRegression uses the Testing set's, so small differences are expected.")
  }
}

# ------------------------------------------------------------------------------------------- result
cat("\nCross-check of the app's analysis in R\n")
cat(sprintf("Data: %s (%d rows); validation: %s; missing values: %s\n\n", S$data_file, S$n_rows, c(A = "Option A (the app's Testing set)", B = "Option B (separate cohort)", C = "Option C (bootstrap, apparent and corrected values)")[S$validation], S$missing))
w <- max(nchar(RES$measure))
cat(sprintf(paste0("%-", w, "s  %-18s  %-18s  %s\n"), "Measure", "App", "R", "How R computed it"))
for (i in seq_len(nrow(RES))) cat(sprintf(paste0("%-", w, "s  %-18s  %-18s  %s\n"), RES$measure[i], RES$app[i], RES$r[i], RES$r_label[i]))
cat("\nHow to read this: with the same patients and the same model, the app and R should agree closely (typically\n")
cat("within 0.01 for AUC, C-index and Brier score). 'Standard' rows use R's usual unpenalised model, so small\n")
cat("differences there are expected. Bootstrap results vary slightly from run to run. A large difference means\n")
cat("the analyses differ: check that the data file and settings are the ones used in the app.\n")
if (length(notes)) { cat("\nNotes:\n"); cat(paste0("- ", notes, collapse = "\n"), "\n") }
invisible(RES)
`;

/**
 * spec: { appVersion, generated, dataFile, externalFile, nRows, outcomeType ('binary' | 'continuous' | 'survival'),
 *   outcome, positive, timeCol, eventCol, eventValue, competingValue, horizon, predictors: [{ name, type, impute,
 *   constant, levels, reference, derivedFrom }], missing ('single' | 'complete' | 'mice'), miceM, validation
 *   ('A' | 'B' | 'C'), testRows (1-based data-row numbers), testCheck { n, events }, B, seed, lambda (the app's L2
 *   penalty for logistic regression), modelName, app: { measure: value | [low, high] } }
 */
function buildRScript(spec) {
    const preds = spec.predictors.map(p => rList({ name: p.name, type: p.type, impute: p.impute || 'median', constant: p.constant ?? null, levels: p.levels || null, reference: p.reference ?? null, derived_from: p.derivedFrom || null }));
    const header = `# =====================================================================================================
# Cross-check in R of an analysis made with the Universal Medical ML/NeuralNet Predictor
# Generated ${spec.generated} by app version ${spec.appVersion}. This file contains no patient data: only the
# analysis settings, the numbers of the rows that formed the Testing set, and the app's results.
#
# How to run: put this script in the same folder as ${rStr(spec.dataFile)}${spec.externalFile ? ` and ${rStr(spec.externalFile)}` : ''},
# then in a terminal:  Rscript ${spec.scriptName || 'crosscheck.R'}   (or open it in RStudio and click Source).
# The analysis is rebuilt independently from the data file with standard R packages, and the app's and R's
# results are printed side by side. Packages used (install once with install.packages(...)):
#   survival, pROC, rms${spec.outcomeType === 'survival' ? ', riskRegression, prodlim' : ''}${spec.missing === 'mice' ? ', mice' : ''}
# =====================================================================================================

S <- list(
  data_file = ${rStr(spec.dataFile)},
  external_file = ${spec.externalFile ? rStr(spec.externalFile) : 'NULL'},
  n_rows = ${rNum(spec.nRows)},
  outcome_type = ${rStr(spec.outcomeType)},
  outcome = ${spec.outcome ? rStr(spec.outcome) : 'NULL'},
  positive = ${spec.positive !== null && spec.positive !== undefined ? rStr(spec.positive) : 'NULL'},
  time_col = ${spec.timeCol ? rStr(spec.timeCol) : 'NULL'},
  event_col = ${spec.eventCol ? rStr(spec.eventCol) : 'NULL'},
  event_value = ${spec.eventValue !== null && spec.eventValue !== undefined ? rStr(spec.eventValue) : 'NULL'},
  competing_value = ${spec.competingValue ? rStr(spec.competingValue) : 'NULL'},
  horizon = ${rNum(spec.horizon)},
  predictors = list(
    ${preds.join(',\n    ')}
  ),
  missing = ${rStr(spec.missing)},
  mice_m = ${rNum(spec.miceM || 5)},
  validation = ${rStr(spec.validation)},
  test_rows = ${spec.validation === 'A' ? rVec(spec.testRows, rNum) : 'NULL'},
  test_check = ${spec.testCheck ? rList(spec.testCheck) : 'NULL'},
  B = ${rNum(spec.B || 200)},
  seed = ${rNum(spec.seed ?? 42)},
  lambda = ${rNum(spec.lambda ?? 0.01)},
  model_name = ${rStr(spec.modelName || '')}
)
APP <- ${rList(spec.app || {})}
`;
    return header + R_CORE;
}



export { buildRScript };

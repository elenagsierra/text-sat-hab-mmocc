def jax_to_numpy_deep(x):
    import jax.numpy as jnp
    import numpy as np

    if isinstance(x, jnp.ndarray):
        return np.array(x)
    elif isinstance(x, dict):
        return {k: jax_to_numpy_deep(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [jax_to_numpy_deep(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(jax_to_numpy_deep(v) for v in x)
    else:
        return x


def fit_biolith(
    features_train,
    features_test,
    y_train,
    y_test,
    regressor_name,
    modalities,
    features_dims,
    regularization="l2",
):

    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro
    from biolith.evaluation import (
        deviance,
        deviance_manual,
        diagnostics,
        log_likelihood_manual,
        lppd,
        lppd_manual,
        posterior_predictive_check,
        residuals,
        waic_manual,
    )
    from biolith.models import occu
    from biolith.regression import BARTRegression, LinearRegression, MLPRegression
    from biolith.utils import fit, fit_multiprocess
    from biolith.utils.grid_search import grid_search_priors
    from numpyro.infer import Predictive
    from numpyro.infer.initialization import init_to_median

    def clear_memory():
        import jax

        try:
            jax.clear_caches()
        except Exception as e:
            print(f"Error clearing JAX caches: {e}")
        finally:
            import gc

            gc.collect()

    biolith_errors = []

    use_cv = True
    use_subprocess = True
    timeout = None

    fit = fit_multiprocess if use_subprocess else fit

    prior_dist = {
        "l1": numpyro.distributions.Laplace(),
        "l2": numpyro.distributions.Normal(),
    }[regularization]

    reg_cls = {
        "LinearRegression": LinearRegression,
        "MLPRegression": MLPRegression,
        "BARTRegression": BARTRegression,
    }[regressor_name]

    features_train_jax = jnp.array(features_train)
    features_test_jax = jnp.array(features_test)
    y_train_jax = jnp.array(y_train)
    y_test_jax = jnp.array(y_test)
    obs_cov_train = jnp.zeros(
        (y_train_jax.shape[0], y_train_jax.shape[1], 0), dtype=features_train_jax.dtype
    )
    obs_cov_test = jnp.zeros(
        (y_test_jax.shape[0], y_test_jax.shape[1], 0), dtype=features_test_jax.dtype
    )

    fit_kwargs = dict(
        init_strategy=init_to_median,
    )

    if use_cv:

        grid_search_results = grid_search_priors(
            occu,
            site_covs=features_train_jax,
            obs_covs=obs_cov_train,
            regressor_occ=reg_cls,
            regressor_det=LinearRegression,
            prior_params_det=False,
            obs=y_train_jax,
            num_chains=1,
            num_samples=500,
            num_warmup=500,
            cv_folds=3,
            timeout=timeout,
            **fit_kwargs,
        )

        results = grid_search_results.best_result
        best_params = grid_search_results.best_params

    else:

        results = fit(
            occu,
            site_covs=features_train_jax,
            obs_covs=obs_cov_train,
            regressor_occ=reg_cls,
            prior_beta=prior_dist,
            prior_alpha=prior_dist,
            obs=y_train_jax,
            num_chains=1,
            num_samples=500,
            num_warmup=500,
            timeout=timeout,
            **fit_kwargs,
        )

        best_params = None

    clear_memory()

    # posterior_samples_numpy = {k: np.array(v) for k, v in results.mcmc.get_samples().items()}

    predictive_train = Predictive(occu, results.mcmc.get_samples())
    posterior_pred_train = predictive_train(
        jax.random.PRNGKey(0),
        site_covs=features_train_jax,
        obs_covs=obs_cov_train,
        regressor_occ=reg_cls,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )

    clear_memory()

    predictive_test = Predictive(occu, results.mcmc.get_samples())
    posterior_pred_test = predictive_test(
        jax.random.PRNGKey(0),
        site_covs=features_test_jax,
        obs_covs=obs_cov_test,
        regressor_occ=reg_cls,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )

    clear_memory()

    residuals_test_occ, residuals_test_det = residuals(posterior_pred_test, y_test)
    mean_absolute_residuals_test_occ = np.mean(np.abs(residuals_test_occ)).item()
    mean_absolute_residuals_test_det = np.nanmean(np.abs(residuals_test_det)).item()

    deviance_train = deviance_manual(
        posterior_pred_train,
        dict(
            site_covs=features_train_jax,
            obs_covs=obs_cov_train,
            obs=y_train_jax,
        ),
    )
    deviance_test = deviance_manual(
        posterior_pred_test,
        dict(
            site_covs=features_test_jax,
            obs_covs=obs_cov_test,
            obs=y_test_jax,
        ),
    )

    clear_memory()

    deviance_numpyro_train = deviance(
        occu,
        posterior_pred_train,
        site_covs=features_train_jax,
        obs_covs=obs_cov_train,
        regressor_occ=reg_cls,
        obs=y_train_jax,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )
    clear_memory()
    deviance_numpyro_test = deviance(
        occu,
        posterior_pred_test,
        site_covs=features_test_jax,
        obs_covs=obs_cov_test,
        regressor_occ=reg_cls,
        obs=y_test_jax,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )

    # Calculate ELPD for test set
    lppd_test = lppd_manual(
        posterior_pred_test,
        dict(
            site_covs=features_test_jax,
            obs_covs=obs_cov_test,
            obs=y_test_jax,
        ),
    )
    ll_test = jnp.nanmean(
        log_likelihood_manual(
            posterior_pred_test,
            dict(
                site_covs=features_test_jax,
                obs_covs=obs_cov_test,
                obs=y_test_jax,
            ),
        )
    ).item()
    waic_test = waic_manual(
        posterior_pred_test,
        dict(
            site_covs=features_test_jax,
            obs_covs=obs_cov_test,
            obs=y_test_jax,
        ),
    )

    # Calculate ELPD for train set
    lppd_train = lppd_manual(
        posterior_pred_train,
        dict(
            site_covs=features_train_jax,
            obs_covs=obs_cov_train,
            obs=y_train_jax,
        ),
    )
    ll_train = jnp.nanmean(
        log_likelihood_manual(
            posterior_pred_train,
            dict(
                site_covs=features_train_jax,
                obs_covs=obs_cov_train,
                obs=y_train_jax,
            ),
        )
    ).item()
    waic_train = waic_manual(
        posterior_pred_train,
        dict(
            site_covs=features_train_jax,
            obs_covs=obs_cov_train,
            obs=y_train_jax,
        ),
    )

    # Calculate ELPD for test set
    lppd_numpyro_test = lppd(
        occu,
        posterior_pred_test,
        site_covs=features_test_jax,
        obs_covs=obs_cov_test,
        regressor_occ=reg_cls,
        obs=y_test_jax,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )

    # Calculate ELPD for train set
    lppd_numpyro_train = lppd(
        occu,
        posterior_pred_train,
        site_covs=features_train_jax,
        obs_covs=obs_cov_train,
        regressor_occ=reg_cls,
        obs=y_train_jax,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )

    clear_memory()

    p_value_test_by_site_freeman_tukey = posterior_predictive_check(
        posterior_pred_test, y_test, "site", "freeman-tukey"
    )
    p_value_test_by_site_chi_squared = posterior_predictive_check(
        posterior_pred_test, y_test, "site", "chi-squared"
    )
    p_value_test_by_revisit_freeman_tukey = posterior_predictive_check(
        posterior_pred_test, y_test, "revisit", "freeman-tukey"
    )
    p_value_test_by_revisit_chi_squared = posterior_predictive_check(
        posterior_pred_test, y_test, "revisit", "chi-squared"
    )

    n_params = sum(
        [
            jnp.prod(jnp.array(samples.shape[1:])).item()
            for key, samples in results.mcmc.get_samples().items()
            if key.startswith("alpha") or key.startswith("beta")
        ]
    )
    n_data_train = jnp.sum(jnp.isfinite(y_test)).item()

    try:
        diagnostics_results = diagnostics(results.mcmc)
    except Exception as e:
        print(f"Error computing diagnostics: {e}")
        diagnostics_results = {}

    # Get feature importance
    feature_importance = None
    if "beta" in results.mcmc.get_samples():
        feature_importance = np.array(
            jnp.abs(results.mcmc.get_samples()["beta"]).mean(axis=0)
        )[
            1:
        ]  # Exclude intercept
    elif "beta_feature_importances" in results.mcmc.get_samples():
        feature_importance = np.array(
            results.mcmc.get_samples()["beta_feature_importances"].mean(axis=0)
        )

    coefficients = None
    if "beta" in results.mcmc.get_samples():
        coefficients = np.array(results.mcmc.get_samples()["beta"]).mean(axis=0)

    try:
        mean_posterior_psi_train = jnp.nanmean(posterior_pred_train["psi"]).item()
        mean_posterior_psi_test = jnp.nanmean(posterior_pred_test["psi"]).item()
    except Exception as e:
        print(f"Error computing mean posterior psi: {e}")
        biolith_errors.append(e)
        mean_posterior_psi_train = np.nan
        mean_posterior_psi_test = np.nan

    try:
        mean_posterior_p_train = jnp.nanmean(
            posterior_pred_train["prob_detection"]
        ).item()
        mean_posterior_p_test = jnp.nanmean(
            posterior_pred_test["prob_detection"]
        ).item()
    except Exception as e:
        print(f"Error computing mean posterior p: {e}")
        biolith_errors.append(e)
        mean_posterior_p_train = np.nan
        mean_posterior_p_test = np.nan

    clear_memory()

    results_null = fit(
        occu,
        site_covs=jnp.zeros((y_train_jax.shape[0], 0), dtype=features_train_jax.dtype),
        obs_covs=obs_cov_train,
        regressor_occ=LinearRegression,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
        obs=y_train_jax,
        num_chains=1,
        num_samples=500,
        num_warmup=500,
        timeout=timeout,
        **fit_kwargs,
    )

    clear_memory()

    results_null_predictive = Predictive(occu, results_null.mcmc.get_samples())
    posterior_pred_null_train = results_null_predictive(
        jax.random.PRNGKey(0),
        site_covs=jnp.zeros(
            (features_train_jax.shape[0], 0), dtype=features_train_jax.dtype
        ),
        obs_covs=obs_cov_train,
        regressor_occ=LinearRegression,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )

    clear_memory()

    posterior_pred_null_test = results_null_predictive(
        jax.random.PRNGKey(0),
        site_covs=jnp.zeros(
            (features_test_jax.shape[0], 0), dtype=features_test_jax.dtype
        ),
        obs_covs=obs_cov_test,
        regressor_occ=LinearRegression,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )

    clear_memory()

    lppd_null_train = lppd_manual(
        posterior_pred_null_train,
        dict(
            site_covs=jnp.zeros(
                (features_train_jax.shape[0], 0), dtype=features_train_jax.dtype
            ),
            obs_covs=obs_cov_train,
            obs=y_train_jax,
        ),
    )
    ll_null_train = jnp.nanmean(
        log_likelihood_manual(
            posterior_pred_null_train,
            dict(
                site_covs=jnp.zeros(
                    (features_train_jax.shape[0], 0), dtype=features_train_jax.dtype
                ),
                obs_covs=obs_cov_train,
                obs=y_train_jax,
            ),
        )
    ).item()
    waic_null_train = waic_manual(
        posterior_pred_null_train,
        dict(
            site_covs=jnp.zeros(
                (features_train_jax.shape[0], 0), dtype=features_train_jax.dtype
            ),
            obs_covs=obs_cov_train,
            obs=y_train_jax,
        ),
    )

    lppd_null_test = lppd_manual(
        posterior_pred_null_test,
        dict(
            site_covs=jnp.zeros(
                (features_test_jax.shape[0], 0), dtype=features_test_jax.dtype
            ),
            obs_covs=obs_cov_test,
            obs=y_test_jax,
        ),
    )
    ll_null_test = jnp.nanmean(
        log_likelihood_manual(
            posterior_pred_null_test,
            dict(
                site_covs=jnp.zeros(
                    (features_test_jax.shape[0], 0), dtype=features_test_jax.dtype
                ),
                obs_covs=obs_cov_test,
                obs=y_test_jax,
            ),
        )
    ).item()
    waic_null_test = waic_manual(
        posterior_pred_null_test,
        dict(
            site_covs=jnp.zeros(
                (features_test_jax.shape[0], 0), dtype=features_test_jax.dtype
            ),
            obs_covs=obs_cov_test,
            obs=y_test_jax,
        ),
    )

    lppd_numpyro_null_train = lppd(
        occu,
        posterior_pred_null_train,
        site_covs=jnp.zeros(
            (features_train_jax.shape[0], 0), dtype=features_train_jax.dtype
        ),
        obs_covs=obs_cov_train,
        regressor_occ=LinearRegression,
        obs=y_train_jax,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )

    lppd_numpyro_null_test = lppd(
        occu,
        posterior_pred_null_test,
        site_covs=jnp.zeros(
            (features_test_jax.shape[0], 0), dtype=features_test_jax.dtype
        ),
        obs_covs=obs_cov_test,
        regressor_occ=LinearRegression,
        obs=y_test_jax,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
    )

    clear_memory()

    results_oracle = fit(
        occu,
        site_covs=jnp.zeros((y_test_jax.shape[0], 0), dtype=features_test_jax.dtype),
        obs_covs=jnp.zeros(
            (y_test_jax.shape[0], y_test_jax.shape[1], 0), dtype=features_test_jax.dtype
        ),
        regressor_occ=LinearRegression,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
        site_random_effects=True,
        obs=y_test_jax,
        num_chains=1,
        num_samples=500,
        num_warmup=500,
        timeout=timeout,
        **fit_kwargs,
    )

    clear_memory()

    predictive_oracle = Predictive(
        occu,
        results_oracle.mcmc.get_samples(),
        return_sites=list(results_oracle.mcmc.get_samples().keys()),
    )
    posterior_pred_oracle_test = predictive_oracle(
        jax.random.PRNGKey(0),
        site_covs=jnp.zeros((y_test.shape[0], 0), dtype=features_test_jax.dtype),
        obs_covs=obs_cov_test,
        regressor_occ=LinearRegression,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
        site_random_effects=True,
    )

    lppd_oracle_test = lppd_manual(
        posterior_pred_oracle_test,
        dict(
            site_covs=jnp.zeros(
                (features_test_jax.shape[0], 0), dtype=features_test_jax.dtype
            ),
            obs_covs=obs_cov_test,
            obs=y_test_jax,
        ),
    )
    ll_oracle_test = jnp.nanmean(
        log_likelihood_manual(
            posterior_pred_oracle_test,
            dict(
                site_covs=jnp.zeros(
                    (features_test_jax.shape[0], 0), dtype=features_test_jax.dtype
                ),
                obs_covs=obs_cov_test,
                obs=y_test_jax,
            ),
        )
    ).item()
    waic_oracle_test = waic_manual(
        posterior_pred_oracle_test,
        dict(
            site_covs=jnp.zeros(
                (features_test_jax.shape[0], 0), dtype=features_test_jax.dtype
            ),
            obs_covs=obs_cov_test,
            obs=y_test_jax,
        ),
    )

    clear_memory()

    lppd_numpyro_oracle_test = lppd(
        occu,
        posterior_pred_oracle_test,
        site_covs=jnp.zeros(
            (features_test_jax.shape[0], 0), dtype=features_test_jax.dtype
        ),
        obs_covs=obs_cov_test,
        regressor_occ=LinearRegression,
        obs=y_test_jax,
        prior_beta=prior_dist,
        prior_alpha=prior_dist,
        site_random_effects=True,
    )

    naive_occupancy_train = (y_train > 0).any(axis=1)
    naive_occupancy_test = (y_test > 0).any(axis=1)

    from sklearn.metrics import average_precision_score, roc_auc_score

    biolith_ap_train = average_precision_score(
        naive_occupancy_train, posterior_pred_train["psi"].mean(axis=0)
    )
    biolith_ap_test = average_precision_score(
        naive_occupancy_test, posterior_pred_test["psi"].mean(axis=0)
    )

    biolith_ap_null_train = average_precision_score(
        naive_occupancy_train, posterior_pred_null_train["psi"].mean(axis=0)
    )
    biolith_ap_null_test = average_precision_score(
        naive_occupancy_test, posterior_pred_null_test["psi"].mean(axis=0)
    )

    biolith_ap_oracle_test = average_precision_score(
        naive_occupancy_test, posterior_pred_oracle_test["psi"].mean(axis=0)
    )
    biolith_ap_oracle_train = float("nan")

    biolith_roc_auc_train = roc_auc_score(
        naive_occupancy_train, posterior_pred_train["psi"].mean(axis=0)
    )
    biolith_roc_auc_test = roc_auc_score(
        naive_occupancy_test, posterior_pred_test["psi"].mean(axis=0)
    )

    biolith_roc_auc_null_train = roc_auc_score(
        naive_occupancy_train, posterior_pred_null_train["psi"].mean(axis=0)
    )
    biolith_roc_auc_null_test = roc_auc_score(
        naive_occupancy_test, posterior_pred_null_test["psi"].mean(axis=0)
    )

    biolith_roc_auc_oracle_test = roc_auc_score(
        naive_occupancy_test, posterior_pred_oracle_test["psi"].mean(axis=0)
    )
    biolith_roc_auc_oracle_train = float("nan")

    modality_coefficients = {}
    offset = 0
    for modality in modalities:
        dim = features_dims[modality]
        coef = coefficients[1:][offset : offset + dim]
        modality_coefficients[modality] = coef
        offset += dim

    results = {
        "regularization": regularization,
        **diagnostics_results,
        "regressor_name": regressor_name,
        "lppd_train": lppd_train,
        "lppd_test": lppd_test,
        "log_likelihood_train": ll_train,
        "log_likelihood_test": ll_test,
        "log_likelihood_null_train": ll_null_train,
        "log_likelihood_null_test": ll_null_test,
        "log_likelihood_oracle_test": ll_oracle_test,
        "waic_train": waic_train,
        "waic_test": waic_test,
        "lppd_numpyro_train": lppd_numpyro_train,
        "lppd_numpyro_test": lppd_numpyro_test,
        "n_params": n_params,
        "n_data_train": n_data_train,
        "mean_absolute_residuals_test_occ": mean_absolute_residuals_test_occ,
        "mean_absolute_residuals_test_det": mean_absolute_residuals_test_det,
        "deviance_train": deviance_train,
        "deviance_test": deviance_test,
        "deviance_numpyro_train": deviance_numpyro_train,
        "deviance_numpyro_test": deviance_numpyro_test,
        "p_value_test_by_site_freeman_tukey": p_value_test_by_site_freeman_tukey,
        "p_value_test_by_site_chi_squared": p_value_test_by_site_chi_squared,
        "p_value_test_by_revisit_freeman_tukey": p_value_test_by_revisit_freeman_tukey,
        "p_value_test_by_revisit_chi_squared": p_value_test_by_revisit_chi_squared,
        "feature_importance": feature_importance,
        "coefficients": coefficients,
        "modality_coefficients": modality_coefficients,
        "mean_posterior_p_train": mean_posterior_p_train,
        "mean_posterior_p_test": mean_posterior_p_test,
        "mean_posterior_psi_train": mean_posterior_psi_train,
        "mean_posterior_psi_test": mean_posterior_psi_test,
        "lppd_null_train": lppd_null_train,
        "lppd_null_test": lppd_null_test,
        "lppd_numpyro_null_train": lppd_numpyro_null_train,
        "lppd_numpyro_null_test": lppd_numpyro_null_test,
        "lppd_oracle_test": lppd_oracle_test,
        "lppd_numpyro_oracle_test": lppd_numpyro_oracle_test,
        "waic_null_train": waic_null_train,
        "waic_null_test": waic_null_test,
        "waic_oracle_test": waic_oracle_test,
        "biolith_ap_train": biolith_ap_train,
        "biolith_ap_test": biolith_ap_test,
        "biolith_ap_null_train": biolith_ap_null_train,
        "biolith_ap_null_test": biolith_ap_null_test,
        "biolith_ap_oracle_train": biolith_ap_oracle_train,
        "biolith_ap_oracle_test": biolith_ap_oracle_test,
        "biolith_roc_auc_train": biolith_roc_auc_train,
        "biolith_roc_auc_test": biolith_roc_auc_test,
        "biolith_roc_auc_null_train": biolith_roc_auc_null_train,
        "biolith_roc_auc_null_test": biolith_roc_auc_null_test,
        "biolith_roc_auc_oracle_train": biolith_roc_auc_oracle_train,
        "biolith_roc_auc_oracle_test": biolith_roc_auc_oracle_test,
        "biolith_errors": biolith_errors,
        "best_params": best_params,
        "mcmc_samples": results.mcmc.get_samples(),
    }

    results = jax_to_numpy_deep(results)

    return results

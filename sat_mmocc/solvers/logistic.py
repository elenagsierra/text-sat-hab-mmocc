from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from mmocc.eval_utils import compute_naive_metrics
from mmocc.utils import cpu_count


def fit_logistic(
    features_train,
    features_test,
    y_train_naive,
    y_test_naive,
    features_dims,
    modalities,
):

    results = {}

    # Hyperparameter grids
    lr_param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["saga"],
    }

    # Cross-validation setup
    cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Logistic Regression with GridSearchCV
    lr_base = LogisticRegression(random_state=0, max_iter=1000)
    lr_grid = GridSearchCV(
        lr_base,
        lr_param_grid,
        cv=cv_folds,
        scoring="average_precision",
        n_jobs=cpu_count(),
    )
    lr_grid.fit(features_train, y_train_naive)
    lr = lr_grid.best_estimator_

    pred_test_lr = lr.predict(features_test)
    pred_train_lr = lr.predict(features_train)
    proba_test_lr = lr.predict_proba(features_test)
    proba_train_lr = lr.predict_proba(features_train)

    results.update(
        {
            f"{k}_test": v
            for k, v in compute_naive_metrics(
                y_test_naive, pred_test_lr, proba_test_lr
            ).items()
        }
    )
    results.update(
        {
            f"{k}_train": v
            for k, v in compute_naive_metrics(
                y_train_naive, pred_train_lr, proba_train_lr
            ).items()
        }
    )

    modality_coefficients = {}
    offset = 0
    for modality in modalities:
        dim = features_dims[modality]
        coef = lr.coef_[0, offset : offset + dim]
        modality_coefficients[modality] = coef
        offset += dim

    results["modality_coefficients"] = modality_coefficients

    return results

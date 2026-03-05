from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_naive_metrics(y_true, y_pred, proba_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    ap = average_precision_score(y_true, proba_pred[:, 1])

    return dict(
        mcc=mcc,
        precision=precision,
        recall=recall,
        f1=f1,
        map=ap,
    )

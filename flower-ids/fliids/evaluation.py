# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Metrics and the three evaluation views (report Section 3.7)."""
# Every metric is computed from a confusion matrix cm, where cm[i, j] counts
# records of true class i predicted as class j. Class 0 is always Benign.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


@torch.no_grad()      # no gradients are needed for prediction (faster, less memory)
def predict(model, x, batch_size=65_536):
    """Class predictions in row order (dropout off, no gradients)."""
    model.eval()      # evaluation mode: dropout switched off
    # score the records in blocks of 65,536 and take, for each record, the class
    # with the highest output score (argmax); returns a NumPy array of class codes
    return torch.cat([model(x[i:i + batch_size]).argmax(1)
                      for i in range(0, len(x), batch_size)]).numpy() if len(x) else np.zeros(0, np.int64)


def confusion(y_true, y_pred, n_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # each (true, predicted) pair gets one number true*C + pred; counting these
    # numbers and reshaping to C x C gives the confusion matrix
    return np.bincount(y_true * n_classes + y_pred,
                       minlength=n_classes ** 2).reshape(n_classes, n_classes)


def confusion_by_client(y_true, y_pred, client, n_clients, n_classes):
    """[K, C, C] confusion counts, one matrix per gateway test portion."""
    # same counting trick with the gateway number added: one C x C matrix per gateway
    idx = (np.asarray(client).astype(np.int64) * n_classes + y_true) * n_classes + y_pred
    return np.bincount(idx, minlength=n_clients * n_classes ** 2).reshape(
        n_clients, n_classes, n_classes)


def metrics(cm, benign_index=0):
    """Accuracy (primary), macro-F1 and per-class scores from confusion counts.

    Precision, recall and F1 are zero when their denominators are zero.
    macro_f1 averages over the complete label set (global and routed views);
    macro_f1_present averages over classes present in the evaluated records
    (gateway-matched view and local validation). FPR treats every attack class
    as malicious and is None when there are no benign records.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)                               # correct predictions per class
    support, predicted = cm.sum(1), cm.sum(0)      # true records / predictions per class
    with np.errstate(divide="ignore", invalid="ignore"):
        # precision: of the records predicted as c, the share that really are c
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        # recall: of the records that really are c, the share that was found
        recall = np.where(support > 0, tp / support, 0.0)
        # F1: balance of precision and recall
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    present = support > 0                          # classes that occur in these records
    benign = cm[benign_index].sum()                # number of benign records
    total = cm.sum()
    return {
        "n": int(total),
        "accuracy": float(tp.sum() / total) if total else None,   # share correct
        "macro_f1": float(f1.mean()),              # mean F1 over all classes
        "macro_f1_present": float(f1[present].mean()) if present.any() else None,
        "precision": precision.round(6).tolist(),
        "recall": recall.round(6).tolist(),
        "f1": f1.round(6).tolist(),
        "classes_present": present.astype(int).tolist(),
        # false-positive rate: benign records predicted as any attack / all benign
        "fpr": float((benign - cm[benign_index, benign_index]) / benign) if benign else None,
    }


def confusion_rows(model, view, cm, alpha=None, client=None):
    """Long-format, non-zero confusion counts for confusion_counts.parquet."""
    # one row per non-zero cell of the matrix, labelled with the model, the view
    # (threat_landscape, gateway_matched or routed), alpha and gateway (-1 = none)
    t, p = np.nonzero(cm)
    return pd.DataFrame({"model": model, "view": view,
                         "alpha": np.nan if alpha is None else float(alpha),
                         "client": -1 if client is None else int(client),
                         "true": t.astype(np.int8), "pred": p.astype(np.int8),
                         "count": cm[t, p].astype(np.int64)})


def save_parquet(df, path):
    # write a table as a compressed Parquet file
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                   compression="zstd")

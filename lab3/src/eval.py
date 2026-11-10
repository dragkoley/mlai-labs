from typing import Any

import numpy as np
from keras import Model
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_recall_curve

from visualization import plot_confusion_matrix, plot_pr_curves, plot_macro_pr_curve


def get_predictions(model: Model, test_dataset: tuple[Any, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_batch, y_pred_batch = [], []

    for x, y in test_dataset:
        y_true_batch.append(y)
        preds = model.predict(x)
        y_pred_batch.append(preds)

    y_true = np.concatenate(y_true_batch, axis=0).flatten().astype(int)
    y_prob = np.concatenate(y_pred_batch, axis=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    return y_true, y_pred, y_prob


def display_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, target_names: list[str]) -> None:
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names)

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    plot_confusion_matrix(display_cm)

    prec_0, rec_0, _ = precision_recall_curve(y_true=(y_true == 0), y_score=(1.0 - y_prob))
    prec_1, rec_1, _ = precision_recall_curve(y_true=(y_true == 1), y_score=y_prob)

    plot_pr_curves(recalls=[rec_0, rec_1], precisions=[prec_0, prec_1], labels=target_names)

    mean_recall = np.linspace(0, 1, 100)
    interp_prec_0 = np.interp(mean_recall, rec_0[::-1], prec_0[::-1])
    interp_prec_1 = np.interp(mean_recall, rec_1[::-1], prec_1[::-1])
    macro_precision = (interp_prec_0 + interp_prec_1) / 2.0

    plot_macro_pr_curve(mean_recall, macro_precision, labels=target_names)

    print("Classification report:")
    print(report)

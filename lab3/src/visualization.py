from typing import Any

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_distribution(df: pd.DataFrame, class_column: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.countplot(data=df, x=class_column)
    for container in ax.containers:
        ax.bar_label(container)
    plt.title(title)
    plt.show()

def plot_confusion_matrix(display: ConfusionMatrixDisplay) -> None:
    fig = plt.figure(figsize=(10, 10))
    display.plot()
    plt.title('Confusion Matrix')
    plt.show()


def plot_pr_curves(recalls: list[Any], precisions: list[Any], labels: list[str]) -> None:
    fig = plt.figure(figsize=(20, 10))

    plt.plot(recalls[0], precisions[0], label=labels[0])
    plt.plot(recalls[1], precisions[1], label=labels[1])
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()


def plot_macro_pr_curve(mean_recall, macro_precision, labels: list[str]) -> None:
    fig = plt.figure(figsize=(20, 10))

    plt.plot(mean_recall, macro_precision, label=f"{labels[0]} & {labels[1]}")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Macro-Average Precision-Recall Curve")
    plt.show()
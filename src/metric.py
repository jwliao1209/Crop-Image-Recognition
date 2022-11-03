import torch
import numpy as np
from sklearn.metrics import confusion_matrix

__all__ = ["compute_acc", "compute_wp_f1"]


def compute_acc(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1)
    acc = (y_pred == y_true).float().mean()
    
    return acc


def compute_wp_f1(y_pred, y_true, eps=1e-20):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    confusion = confusion_matrix(y_true, y_pred)

    f1_dict = {}
    precision_list, TP_list, FN_list = [], [], []

    for i in range(len(confusion)):
        TP = confusion[i, i]
        FP = sum(confusion[:, i]) - TP
        FN = sum(confusion[i, :]) - TP

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        result_f1 = (2 * precision  * recall + eps) / (precision + recall + eps)

        TP_list.append(TP)
        FN_list.append(FN)

        f1_dict[i] = result_f1
        precision_list.append(precision)

    total_image = y_pred.shape[0]
    weighted = 0.

    for i in range(len(confusion)):
        weighted += precision_list[i] * (TP_list[i] + FN_list[i])

    WP = weighted / total_image

    return f1_dict, WP

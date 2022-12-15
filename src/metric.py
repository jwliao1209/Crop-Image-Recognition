import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_acc(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1)
    acc = (y_pred == y_true).float().mean()
    
    return acc


class Evaluator():
    def __init__(self):
        self.y_preds = []
        self.y_trues = []
        self.num_classes = 0
        self.eps = 1e-20
    
    def reset(self):
        self.y_preds = []
        self.y_trues = []
        return

    def add(self, y_pred, y_true):
        y_pred = y_pred.argmax(dim=1)
        self.y_preds.extend(y_pred.tolist())
        self.y_trues.extend(y_true.tolist())

        return

    def get_confusion_matrix(self):
        y_preds = np.array(self.y_preds)
        y_trues = np.array(self.y_trues)
        self.confusion_mtx = confusion_matrix(y_trues, y_preds)
        self.num_classes = len(np.unique(y_trues))

        if len(self.confusion_mtx) != self.num_classes:
            raise ValueError(f"The confusion matrix does not complete.")

        return self.confusion_mtx
    
    def compute_f1_scroe(self, precision, recall):
        return (2 * precision * recall + self.eps) / (precision + recall + self.eps)

    def compute_f1_dict(self):
        self.f1_dict = {}
        tp_list, fn_list, precision_list = [], [], []
        confusion_mtx = self.get_confusion_matrix()
        for i in range(len(confusion_mtx)):
            tp = confusion_mtx[i, i]
            fp = sum(confusion_mtx[:, i]) - tp
            fn = sum(confusion_mtx[i, :]) - tp

            precision = (tp + self.eps) / (tp + fp + self.eps)
            recall    = (tp + self.eps) / (tp + fn + self.eps)
            f1_score  = self.compute_f1_scroe(precision, recall)

            self.f1_dict[i] = f1_score
            tp_list.append(tp)
            fn_list.append(fn)
            precision_list.append(precision)

        return self.f1_dict, tp_list, fn_list, precision_list
    
    def compute_wp(self):
        _, tps, fns, precisions = self.compute_f1_dict()
        num = len(self.y_preds)
        weighted_sum = sum([precisions[i] * (tps[i] + fns[i])
                            for i in range(self.num_classes)])
        self.wp = weighted_sum / num

        return self.wp
    
    def get_min_f1(self):
        return min(self.f1_dict.values())

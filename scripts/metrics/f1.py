import torch
from sklearn.metrics import confusion_matrix

def get_wp_f1(y_pred, y_true):
    """ Precision_Recall_F1score metrics
    y_pred: the predicted score of each class, shape: (Batch_size, num_classes)
    y_true: the ground truth labels, shape: (Batch_size,) for 'multi-class' or (Batch_size, n_classes) for 'multi-label'
    """
    eps=1e-20
    # y_pred = torch.argmax(y_pred, dim=1)
    if torch.is_tensor(y_pred)==True and torch.is_tensor(y_true)==True:
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()


    # F1_sci = f1_score(y_true, y_pred, average=None)
    confusion = confusion_matrix(y_true, y_pred)

    f1_dict = {}
    precision_list = []
    TP_list = []
    FN_list = []
    for i in range(len(confusion)):
        TP = confusion[i, i]
        FP = sum(confusion[:, i]) - TP
        FN = sum(confusion[i, :]) - TP

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        result_f1 = 2 * precision  * recall / (precision + recall + eps)

        TP_list.append(TP)
        FN_list.append(FN)
        f1_dict[i] = result_f1
        precision_list.append(precision)

    total_image = len(y_pred)
    weighted = 0.
    for i in range(len(confusion)):
        weighted += precision_list[i] * (TP_list[i] + FN_list[i])

    WP = weighted / total_image

    return f1_dict, WP
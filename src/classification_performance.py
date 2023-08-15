from typing import Any, Dict, List, Optional

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np


class ClassificationPerformance:
    """
    Classification performance metrics are used to evaluate binary classification results.

    Further information: https://scikit-learn.org/0.15/modules/model_evaluation.html
    """

    def __init__(self):
        pass

    @staticmethod
    def accuracy_score(y_true: List[float], y_pred: List[float]):
        """ Accuracy = (TP + TN) / (TP + FP + FN + TN) """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy_score = (tp + tn) / (tp + fp + fn + tn)
        return accuracy_score

    @staticmethod
    def sensitivity(y_true: List[float], y_pred: List[float]):
        """ Sensitivity / Recall = TP / (TP + FN)
        Out of all the people that have the disease, how many got positive test results?
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        return sensitivity

    @staticmethod
    def specificity(y_true: List[float], y_pred: List[float]):
        """ Specificity = TN / (TN + FP)
        Out of all the people that do not have the disease, how many got negative results?
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        return specificity

    @staticmethod
    def precision(y_true: List[float], y_pred: List[float]):
        """ Precision = TP / (TP + FP)
        Out of all the examples that predicted as positive, how many are really positive?
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp)
        return precision

    @staticmethod
    def auc(y_true: List[float], y_scores: List[float]):
        """ Area Under the Curve (AUC) from prediction scores """
        return roc_auc_score(y_true, y_scores, average=None)

    @staticmethod
    def confusion_matrix(y_true: List[float], y_pred: List[float], plot_mode: bool = False):
        """ Confusion matrix: TN, FP, FN, TP """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        if plot_mode is True:
            plt.matshow(cm)
            plt.title('Confusion matrix')
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

        return float(tn), float(fp), float(fn), float(tp)

    @staticmethod
    def generate_predictions_and_labels(preds: Dict[str, Any],
                                        labels: List[Dict[str, Any]],
                                        pos_threshold_num: int = None,
                                        op_shift: float = 0.0):
        """
        Generate predictions and labels by comparing the classification results and
        the labels from the public dataset metadata.
        Return the list of predicted labels, true labels, and predicted scores.

        Predictions are a list of dictionaries per volume.
        Each dictionary of predictions has the key "SeriesInstanceUID", and a list of predicted probabilities per slab.
        E.g. [{'1.2.3.4.1': [p_benign, p_malignant], [0.3, 0.7], [0.1, 0.9]}, {'1.2.3.4.2': [0.2, 0.8]}, ... ]

        :param preds: list of dictionaries per volume, which contain predictions per slab
        :param labels: list of series with labels (0/1)
        :param pos_threshold_num: the number to choose the top results
        :param op_shift:
        :return: predicted labels, true labels, and predicted probabilities
        """

        if pos_threshold_num is None:
            pos_threshold_num = 1

        # Create a dictionary to store the number of slabs for each SeriesInstanceUID
        slab_nums = {uid: len(preds[uid]) for uid in preds}

        # List of SeriesInstanceUIDs from predictions
        series_instance_uids = [key for key, value in preds.items()]

        y_pred = []
        y_true = []
        y_scores = []

        for uid in series_instance_uids:

            slab_num = slab_nums[uid]

            series = [item for item in labels if item['SeriesInstanceUID'] == uid]
            if len(series) != 0:

                # True label
                true_label = series[0]['Class']
                y_true.append(true_label)

                # List of predicted probabilities for the current SeriesInstanceUID
                pred = np.asarray(preds[uid])
                pred = pred + [[op_shift, -op_shift]] * len(pred)
                prob_benign = pred[:, 0]
                prob_malignant = pred[:, 1]

                # Predicted label
                pred_label = 0
                # Check for top 1/2/3 predictions
                malignant_num = np.sum(prob_malignant > prob_benign)
                if malignant_num >= pos_threshold_num or malignant_num == slab_num:
                    pred_label = 1
                y_pred.append(pred_label)

                # Predicted probability
                # Take the average of top 1/2/3 probabilities for positive class
                prob_malignant = np.sort(prob_malignant)
                prob_malignant = np.mean(prob_malignant[-pos_threshold_num:])
                y_scores.append(prob_malignant)

        return y_pred, y_true, y_scores

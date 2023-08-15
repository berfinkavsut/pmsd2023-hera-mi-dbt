from program.program import Program
from src.classification_performance import ClassificationPerformance as cp

from typing import List, Dict, Any

import pandas as pd

import os
import json


class PerformanceMetrics(Program):
    """
    PerformanceMetrics is used to read classification results and compute binary classification metrics.
    """

    def read_prediction_results(self, config_hash_code: str):
        """
        Read prediction results and return it with the list of SeriesInstanceUIDs.

        The classification results are provided in a JSON file as a dictionary of list of predictions.
        The file has the results as follows:  {'seriesUID': [predictions per slab]}
        E.g. [{'1.2.3.4.1': [p_benign, p_malignant], [0.3, 0.7], [0.1, 0.9]}, {'1.2.3.4.2': [0.2, 0.8]}, ... ]
        """

        pred_dir = os.path.join('', *[self._output_data_dir, 'predictions'])
        pred_filepath = os.path.join(pred_dir, f'predictions_{config_hash_code}.json')

        if not os.path.exists(pred_filepath):
            return None

        with open(pred_filepath, 'r') as file:
            pred_results = json.load(file)

        # Retrieve the list of SeriesInstanceUIDs from results
        # Later, use it to filter the dataset with SeriesInstanceUIDs in order to create true labels
        series_instance_uids = [key for key, value in pred_results.items()]

        return pred_results, series_instance_uids

    def generate_predictions_and_labels(self, preds: Dict[str, Any],
                                        labels: List[Dict[str, Any]],
                                        pos_threshold_num: int = None,
                                        op_shift: float = 0.0):
        """
        Generate predictions and labels by comparing the classification results and
        the labels from the public dataset metadata.
        """
        y_pred, y_true, y_scores = cp.generate_predictions_and_labels(preds, labels, pos_threshold_num,
                                                                      op_shift=op_shift)
        return y_pred, y_true, y_scores

    def compute_performance_metrics(self, config_hash_code: str,
                                    y_pred: List[float],
                                    y_true: List[float],
                                    y_scores: List[float],
                                    top_num: int,
                                    op_shift=0.0):
        """
        Compute the classification performance metrics and
        save them in a JSON file inside output/performance_metrics/{top_num}.
        """

        performance_metrics_dir = os.path.join('', *[self._output_data_dir, 'performance_metrics', f'top_{top_num}'])
        os.makedirs(performance_metrics_dir, exist_ok=True)

        performance_metrics_dict = {}
        performance_metrics_dict['top'] = top_num
        performance_metrics_dict['op_shift'] = op_shift

        for performance_metric in self._performance_metrics:
            if performance_metric == 'sensitivity':
                performance_metrics_dict['sensitivity'] = cp.sensitivity(y_true, y_pred)
            elif performance_metric == 'specificity':
                performance_metrics_dict['specificity'] = cp.specificity(y_true, y_pred)
            elif performance_metric == 'precision':
                performance_metrics_dict['precision'] = cp.precision(y_true, y_pred)
            elif performance_metric == 'accuracy_score':
                performance_metrics_dict['accuracy_score'] = cp.accuracy_score(y_true, y_pred)
            elif performance_metric == 'auc':
                performance_metrics_dict['auc'] = cp.auc(y_true, y_scores)
            elif performance_metric == 'youden':
                performance_metrics_dict['youden'] = cp.specificity(y_true, y_pred) + cp.sensitivity(y_true, y_pred) - 1

        tn, fp, fn, tp = cp.confusion_matrix(y_true, y_pred)
        performance_metrics_dict['tn'] = tn
        performance_metrics_dict['fp'] = fp
        performance_metrics_dict['fn'] = fn
        performance_metrics_dict['tp'] = tp

        performance_metrics_filepath = os.path.join('', *[performance_metrics_dir,
                                                          f'performance_metrics_{config_hash_code}.json'])
        with open(performance_metrics_filepath, 'w') as file:
            json.dump(performance_metrics_dict, file, indent=4)

        return performance_metrics_dict

    def get_performance_metrics(self, config_hash_code: str, performance_metrics: Dict[str, float]):
        """
        Return the classification performance metrics dictionary to be saved inside the CSV file later.
        """

        result_dict = {'config_hash_code': config_hash_code}
        config = self.read_config_json(config_hash_code=config_hash_code)
        for elem in config:
            if elem == 'thickness_overlap':
                result_dict['thickness'] = config[elem][0]
                result_dict['overlap'] = config[elem][1]
            elif elem == 'slice_skip_ratio':
                result_dict['slice_skip_top'] = config[elem][0]
                result_dict['slice_skip_bottom'] = config[elem][1]
            else:
                result_dict[elem] = config[elem]

        result_dict.update(performance_metrics)

        return result_dict

    def save_performance_metrics_to_csv(self, results: List[Dict[str, Any]], csv_filename: str, top_num: int = None):
        """
        Save the CSV file with classification performance metrics.
        If the top number is provided, then save the CSV file inside output/performance_metrics/{top_num}.
        """

        df_results = pd.DataFrame(results)

        if top_num is not None:
            subdir = f'top_{top_num}'
            performance_metrics_dir = os.path.join('', *[self._output_data_dir, 'performance_metrics', subdir])
        else:
            performance_metrics_dir = os.path.join('', *[self._output_data_dir, 'performance_metrics'])

        os.makedirs(performance_metrics_dir, exist_ok=True)
        csv_path = os.path.join('', *[performance_metrics_dir, csv_filename])
        df_results.to_csv(csv_path, index=False)

        return df_results

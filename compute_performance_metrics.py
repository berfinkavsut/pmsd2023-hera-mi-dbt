from program.arg_parser import *
from program.performance_metrics import PerformanceMetrics

args = parse_arguments()
dataset_config = load_dataset(args)
settings_config = load_settings(args)

perf_metrics = PerformanceMetrics(settings_config=settings_config, dataset_config=dataset_config)
perf_metrics.init()

# List config hash codes inside the output folder
config_hash_codes = perf_metrics.list_config_hash_codes()

full_results = []
for op_shift in settings_config['op-shifts']:
    for pos_threshold_num in settings_config['pos-threshold-num']:

        # Compute classification performance metrics for each slab generation configuration
        results = []
        for config_hash_code in config_hash_codes:

            predictions = perf_metrics.read_prediction_results(config_hash_code=config_hash_code)

            if predictions is None:
                continue

            [preds, series_instance_uids] = predictions

            # Labels: cancer = 1 | normal or benign = 0
            labels = perf_metrics.filter_series_by_labels(labels=['normal', 'benign', 'cancer'],
                                                          pos_labels=['cancer'],
                                                          series_instance_uids=series_instance_uids)
            labels = labels.to_dict('records')

            # Generate the list of predictions and labels
            y_pred, y_true, y_scores = perf_metrics.generate_predictions_and_labels(preds, labels, pos_threshold_num, op_shift=op_shift)

            performance_metrics_dict = perf_metrics.compute_performance_metrics(config_hash_code=config_hash_code,
                                                                                y_pred=y_pred,
                                                                                y_true=y_true,
                                                                                y_scores=y_scores,
                                                                                top_num=pos_threshold_num,
                                                                                op_shift=op_shift)

            result_dict = perf_metrics.get_performance_metrics(config_hash_code=config_hash_code,
                                                               performance_metrics=performance_metrics_dict)

            results.append(result_dict)

        full_results.extend(results)

        perf_metrics.save_performance_metrics_to_csv(results=results,
                                                     csv_filename=f"performance_metrics_op{str(op_shift).replace('.', '')}_top{pos_threshold_num}.csv",
                                                     top_num=pos_threshold_num)

perf_metrics.save_performance_metrics_to_csv(results=full_results,
                                             csv_filename=f"performance_metrics.csv")

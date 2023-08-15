from program.iqa_metrics import IQAMetrics
from program.arg_parser import *


args = parse_arguments()
dataset_config = load_dataset(args)
settings_config = load_settings(args)

iqa_metrics = IQAMetrics(settings_config=settings_config, dataset_config=dataset_config)
iqa_metrics.init()

# List config hash codes from the output folder
config_hash_codes = iqa_metrics.list_config_hash_codes()

# Compute IQA metrics for each slab generation configuration
results = []
for config_hash_code in config_hash_codes:

    series_instance_uids = iqa_metrics.list_series_instance_uids(config_hash_code=config_hash_code)

    # Only filter the series that we can compute IQA metrics, i.e. CNR and contrast
    # CNR and contrast are computed only for the DBT volumes having 'boxes' information
    df_series = iqa_metrics.filter_series(dataset='boxes', labels=None, series_instance_uids=series_instance_uids)
    series_instance_uids = df_series['SeriesInstanceUID'].tolist()

    if len(series_instance_uids) == 0:
        continue

    # Compute IQA metrics for each volume
    for series_instance_uid in series_instance_uids:

        slabs = iqa_metrics.read_slabs(config_hash_code=config_hash_code,
                                       series_instance_uid=series_instance_uid)

        if slabs is None:
            continue

        # Plots with the boxes for pathological regions
        # iqa_metrics.draw_box_slab(image=slabs,
        #                           config_hash_code=config_hash_code,
        #                           series_instance_uid=series_instance_uid)

        iqa_metrics_dict = iqa_metrics.compute_iqa_metrics(slabs=slabs,
                                                           config_hash_code=config_hash_code,
                                                           series_instance_uid=series_instance_uid,
                                                           )

        # If IQA metrics were already computed, read them from slab folders
        # iqa_metrics_dict = iqa_metrics.read_iqa_metrics(config_hash_code=config_hash_code,
        #                                                 series_instance_uid=series_instance_uid)

        result_dict = iqa_metrics.get_iqa_metrics(config_hash_code=config_hash_code,
                                                  series_instance_uid=series_instance_uid,
                                                  iqa_metrics=iqa_metrics_dict)
        results.append(result_dict)

iqa_metrics.save_iqa_metrics_to_csv(results=results, csv_filename='iqa_metrics.csv')

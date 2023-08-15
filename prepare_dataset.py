from src.dataset import Dataset
from program.arg_parser import *


args = parse_arguments()
dataset_config = load_dataset(args)

# Dataset
dataset = Dataset(config=dataset_config)
series_list = dataset.filter_series(dataset="boxes")
series_list = series_list.to_dict('records')
series_list = series_list[0:dataset_config['dicom-file-number']]

# Get series with a certain SeriesInstanceUID
# series_instance_uid = '1.2.826.0.1.3680043.8.498.12015914516035206099641097291298835020'
# series_list = dataset.get_series_by_uid(series_instance_uid=series_instance_uid).to_dict('records')

# Download DICOM files
dataset.download_all_dicom(series_list=series_list)

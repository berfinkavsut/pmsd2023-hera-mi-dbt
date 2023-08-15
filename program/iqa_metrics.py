from program.slab_generation import SlabGeneration
from src.dicom_file_processor import DICOMFileProcessor as dfp
from src.image_quality_assessment import IQA as iqa

from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import json


class IQAMetrics(SlabGeneration):
    """
    IQAMetrics is used to compute IQA metrics which were selected in the config/settings.yaml.
    Currently, only CNR and contrast are computed.
    Hence, the public dataset metadata is necessary to get pathological region coordinates.

    """

    def compute_iqa_metrics(self, slabs: np.ndarray, config_hash_code: str, series_instance_uid: str):
        """
        Compute the IQA metrics for slabs and save the IQA metrics
        inside output/{config_hash_code}/{series_instance_uid}.
        """

        iqa_metrics_dict = {}

        boxes = self.get_box_slice(image=slabs, series_instance_uid=series_instance_uid)
        if boxes is None:
            iqa_metrics_dict = {}
        else:
            # Convert from slice index to slab index
            config = self.read_config_json(config_hash_code=config_hash_code)
            slab_indices = self._compute_slab_indices(config, slabs.shape[0])

            # Boxes: 'slab', list of 'x' indices, list of 'y' indices
            boxes = self._convert_slice_to_slab_indices(boxes, slab_indices)

            for iqa_metric in self._iqa_metrics:
                if iqa_metric == 'cnr' and boxes is not None:
                    cnr_values, mean_cnr, std_cnr = iqa.contrast_to_noise_ratio_3d(image=slabs,
                                                                                   object_coords=boxes,
                                                                                   plot_mode=False)
                    iqa_metrics_dict['cnr'] = {'values': cnr_values,
                                               'mean': mean_cnr,
                                               'std': std_cnr}

                elif iqa_metric == 'contrast' and boxes is not None:
                    contrast_values, mean_contrast, std_contrast = iqa.contrast_3d(image=slabs,
                                                                                   object_coords=boxes,
                                                                                   plot_mode=False)
                    iqa_metrics_dict['contrast'] = {'values': contrast_values,
                                                    'mean': mean_contrast,
                                                    'std': std_contrast}

        iqa_metrics_dir = os.path.join('', *[self._output_data_dir, 'iqa_metrics', config_hash_code])
        os.makedirs(iqa_metrics_dir, exist_ok=True)
        iqa_filepath = os.path.join(iqa_metrics_dir, f'iqa_metrics_{series_instance_uid}.json')

        with open(iqa_filepath, 'w') as file:
            json.dump(iqa_metrics_dict, file, indent=2)

        return iqa_metrics_dict

    def read_iqa_metrics(self, config_hash_code: str, series_instance_uid: str):
        """
        Read the IQA metrics, which were stored in the JSON files inside the slab folders.
        Each configuration folder contains the slab generation results for each DICOM SeriesInstanceUID.
        """

        # Get the IQA metrics filepath
        iqa_metrics_dir = os.path.join('', *[self._output_data_dir, 'iqa_metrics', config_hash_code])
        iqa_filepath = os.path.join(iqa_metrics_dir, f'iqa_metrics_{series_instance_uid}.json')

        with open(iqa_filepath, 'r') as file:
            iqa_metrics_dict = json.load(file)

        return iqa_metrics_dict

    def get_iqa_metrics(self, config_hash_code: str, series_instance_uid: str, iqa_metrics: Dict[str, float]):
        """
        Return the IQA metrics dictionary to be saved inside the CSV file later.
        """

        result_dict = {'series_instance_uid': series_instance_uid,
                       'config_hash_code': config_hash_code}

        config = self.read_config_json(config_hash_code=config_hash_code)
        for elem in config:
            if elem == 'thickness_overlap':
                result_dict['thickness'] = config[elem][0]
                result_dict['overlap'] = config[elem][1]
            else:
                result_dict[elem] = config[elem]

        # Save IQA metrics in each column as (mean, std) pairs
        for iqa_metric in iqa_metrics:
            result_dict[f'{iqa_metric}_mean'] = iqa_metrics[iqa_metric]['mean']
            result_dict[f'{iqa_metric}_std'] = iqa_metrics[iqa_metric]['std']

        return result_dict

    def save_iqa_metrics_to_csv(self, results: List[Dict[str, Any]], csv_filename):
        """
        Save the CSV file with IQA metrics after sorting by SeriesInstanceUIDs.
        """

        # Sort results by Series Instance UIDs
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='series_instance_uid')

        # Save CSV file
        csv_path = os.path.join('', *[self._output_data_dir, 'iqa_metrics', csv_filename])
        df_results.to_csv(csv_path, index=False)

        return df_results

    def read_iqa_metrics_csv(self, csv_filename: str):
        """
        Read the CSV file with IQA metrics.
        """

        # Read CSV file
        csv_path = os.path.join('', *[self._output_data_dir, 'iqa_metrics', csv_filename])
        df_results = pd.read_csv(csv_path)

        return df_results

    def get_box_slice(self, image: np.ndarray, series_instance_uid: str):
        """
        Read the boxes information from the data frame `df_series.
        If the view laterality of image was stored wrongly, flip the box coordinates.
        Return the boxes with the start and end coordinates instead of `width` and `height.
        """

        # If no dataset was provided, there is no box information
        boxes = None
        if self._df_series is not None:

            # If SeriesInstanceUID is not from the dataset, there is no box information
            series = self._df_series[self._df_series['SeriesInstanceUID'] == series_instance_uid]
            if len(series) == 0:
                boxes = None

            else:
                series = series.iloc[0].to_dict()
                view_laterality = str(series['View'])[0]
                image_laterality = dfp.is_image_laterality_correct(image, view_laterality)

                # If SeriesInstanceUID was not in the list of boxes, there is no box information
                boxes_ = self._dataset.get_boxes_param(series_instance_uid)
                if boxes_ is None:
                    return None

                boxes = []

                x_dim = image.shape[2]
                y_dim = image.shape[1]

                for i in range(len(boxes_)):
                    box = boxes_[i]

                    # x in the horizontal axis from left to right
                    # y in the vertical axis from up to down
                    slice_idx = box['Slice']
                    x = box['X']
                    y = box['Y']
                    width = box['Width']
                    height = box['Height']

                    x = min(max(x, 0), x_dim - 1)
                    y = min(max(y, 0), y_dim - 1)

                    # If left-right image laterality is wrong,
                    # flip the box coordinates in x-axis
                    if image_laterality is False:
                        x = x_dim - x - width

                    x1 = x; x2 = x + width
                    y1 = y; y2 = y + height

                    boxes.append({'slice': slice_idx, 'x': [x1, x2], 'y': [y1, y2]})

        return boxes

    def draw_box_slice(self, image: np.ndarray, series_instance_uid: str, line_width: int = 4, plot_mode: bool = True):
        """
        Draw boxes for each slice having boxes and plot the slice images if desired.
        """

        boxes = self.get_box_slice(image, series_instance_uid)

        if boxes is None:
            return None

        box_num = len(boxes)
        x_dim = image.shape[2]
        y_dim = image.shape[1]

        image_set = np.zeros(shape=(box_num, y_dim, x_dim))
        slices = []

        for i in range(box_num):
            box = boxes[i]

            # x in the horizontal axis from left to right
            # y in the vertical axis from up to down
            slice_idx = box['slice']
            x = box['x']
            y = box['y']

            slice_image = image[slice_idx, :, :].copy()
            slice_image[y[0]: y[0] + line_width, x[0]: x[1]] = 1
            slice_image[y[1] - line_width: y[1], x[0]: x[1]] = 1
            slice_image[y[0]: y[1], x[0]: x[0] + line_width] = 1
            slice_image[y[0]: y[1], x[1] - line_width: x[1]] = 1

            image_set[i, :, :] = slice_image.copy()

            slices.append(slice_idx)

        if plot_mode is True:
            self._plot_box(image_set, slices)

        return image_set, slices

    def draw_box_slab(self, image: np.ndarray, config_hash_code: str, series_instance_uid: str, line_width: int = 4, plot_mode: bool = True):
        """
        Draw boxes for each slab having boxes and plot the slab images if desired.
        """

        boxes = self.get_box_slice(image, series_instance_uid)

        if boxes is None:
            return None

        # Convert from slice to slab indices
        config = self.read_config_json(config_hash_code=config_hash_code)
        slab_indices = self._compute_slab_indices(config, image.shape[0])
        boxes = self._convert_slice_to_slab_indices(boxes, slab_indices)

        slabs_with_box_num = len(boxes)
        x_dim = image.shape[2]
        y_dim = image.shape[1]

        image_set = np.zeros(shape=(slabs_with_box_num, y_dim, x_dim))
        slabs = []

        for i in range(slabs_with_box_num):
            box = boxes[i]

            # x in the horizontal axis from left to right
            # y in the vertical axis from up to down
            slab_idx = box['slab']
            x_ = box['x']
            y_ = box['y']

            box_num = len(x_)
            slab_image = image[slab_idx, :, :].copy()

            for j in range(box_num):
                x = x_[j]; y = y_[j]
                slab_image[y[0]: y[0] + line_width, x[0]: x[1]] = 1
                slab_image[y[1] - line_width: y[1], x[0]: x[1]] = 1
                slab_image[y[0]: y[1], x[0]: x[0] + line_width] = 1
                slab_image[y[0]: y[1], x[1] - line_width: x[1]] = 1

            image_set[i, :, :] = slab_image.copy()
            slabs.append(slab_idx)

        if plot_mode is True:
            self._plot_box(image_set, slabs, title='Slab')

    def _convert_slice_to_slab_indices(self, boxes: List[Dict[str, Any]], slab_indices: Dict[int, Tuple[int, int]]):
        """
        Convert slice indices to slab indices by checking the index range of each slab.

        If there is any slice having a box inside the range of slabs,
        then the slabs is considered as having the same box as well.
        Slice indices are taken from boxes and slab indices are given as an argument.
        """

        boxes_new = []
        for i in range(len(boxes)):
            slice_idx = boxes[i]['slice']

            for j in range(len(slab_indices)):

                if slab_indices[j][0] <= slice_idx < slab_indices[j][1]:  # last index is not included
                    boxes[i]['slab'] = j
                    boxes_new.append(boxes[i].copy())

        # Delete slice keys
        boxes_new = [{key: value for key, value in box.items() if key != 'slice'} for box in boxes_new]

        grouped_boxes = {}
        for box in boxes_new:
            slab_val = box['slab']
            if slab_val in grouped_boxes:
                grouped_boxes[slab_val]['x'].append(box['x'])
                grouped_boxes[slab_val]['y'].append(box['y'])
            else:
                grouped_boxes[slab_val] = {'x': [box['x']], 'y': [box['y']], 'slab': slab_val}

        result = list(grouped_boxes.values())
        return result

    def _compute_slab_indices(self, config: Dict[str, Any], slab_num: int):
        """
        Compute slab indices from thickness and overlap.
        The limit is defined by the slab number.
        """

        # Get thickness and overlap
        thickness, overlap = config['thickness_overlap']

        # Compute slab indices
        slab_indices = {}
        i = 0
        for idx in range(slab_num):
            j = i + thickness  # not included
            slab_indices[idx] = (i, j)
            i = j - overlap

        return slab_indices

    def _plot_box(self, image_set: np.ndarray, slices: List[int], title: str = 'Slice'):
        for i in range(len(image_set)):
            image = image_set[i, :, :]
            slice_idx = slices[i]
            plt.title(f"{title} #{slice_idx}")
            plt.imshow(image, cmap='gray')
            plt.show()

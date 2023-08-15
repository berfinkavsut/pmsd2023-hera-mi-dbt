import io
import os
import requests
import pandas as pd

from typing import Any, Dict, List
from zipfile import ZipFile


class Dataset:
    """
    Dataset class is used to download DICOM files from Cancer Imaging Archive by using the NBIA Data Retriever
    and process the information for series by using CSV files.

    Database: Cancer Imaging Archive / Breast Cancer Screening - Digital Breast Tomosynthesis
    Source: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2783046
    Link: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580

    There are three CSV files that you need for creating the dataset: train, boxes, and labels.

    Train dataset has these keys: PatientID,StudyUID,View,descriptive_path,classic_path
    Label dataset has these keys: PatientID,StudyUID,View,Normal,Actionable,Benign,Cancer
    Boxes dataset has these keys: PatientID,StudyUID,View,Subject,Slice,X,Y,Width,Height,Class,AD,VolumeSlices

    """

    def __init__(self, config: Dict[str, Any]):

        # URL to download DICOM files
        self._base_url = config['base-url']

        # Directory to save DICOM files
        self._data_dir = config['data-dir']

        # CSV files for train dataset
        self._train_filepath = config['dataset-train-filepath']
        self._boxes_train_filepath = config['dataset-boxes-train-filepath']
        self._labels_train_filepath = config['dataset-labels-train-filepath']

        # Train dataset series
        df_train = pd.read_csv(self._train_filepath)
        df_labels = pd.read_csv(self._labels_train_filepath)

        # Merge train dataset series with their labels
        self._primary_key = ['PatientID', 'StudyUID', 'View']
        self._df_series = pd.merge(df_labels, df_train, on=self._primary_key)

        # Train dataset with boxes
        self._boxes_key = ['Slice', 'X', 'Y', 'Width', 'Height']
        self._df_boxes = pd.read_csv(self._boxes_train_filepath)

        # Preprocess for labels
        self._preprocess()

    def _preprocess(self):
        """
        Convert one-hot encoding representation of labels to strings.
        Labels: 'normal', 'actionable', 'benign', 'cancer'.

        :return: None
        """
        classes = ['Normal', 'Actionable', 'Benign', 'Cancer']
        self._df_series['Class'] = self._df_series[classes].idxmax(axis=1).apply(lambda x: x.lower())
        self._df_series.drop(['Normal', 'Actionable', 'Benign', 'Cancer'], axis=1, inplace=True)

        # Add column for SeriesInstanceUID from classic_path
        self._df_series['SeriesInstanceUID'] = self._df_series['classic_path'].apply(self._extract_series_instance_uid)

    def filter_series(self, dataset: str = 'train', labels: List[str] = None, series_instance_uids: List[str] = None):
        """
        Retrieve the list of patients from the selected dataset: train dataset with/without box information.
        Filter the series with labels: 'normal', 'actionable', 'benign', 'cancer'.

        :param dataset: train set with/without boxes
        :param labels: 'normal', 'actionable', 'benign', or 'cancer'
        :param series_instance_uids: list of series unique ids for DICOM files
        :return: data frame of series
        """

        df_series = self._df_series  # train & label

        # Filter with series having box information
        if dataset == 'boxes':
            df_boxes = pd.read_csv(self._boxes_train_filepath)
            df_boxes = df_boxes.loc[:, self._primary_key]
            df_boxes = df_boxes.drop_duplicates()
            df_series = pd.merge(df_boxes, self._df_series, on=self._primary_key)  # train & label & boxes

        # Filter by series instance uids
        if series_instance_uids is not None:
            series_instance_uids = set(series_instance_uids)
            df_series = df_series[df_series['SeriesInstanceUID'].isin(series_instance_uids)]

        # Filter by label
        if labels is not None:
            labels = set(labels)
            df_series = df_series[df_series['Class'].isin(labels)]

        return df_series

    def get_series_by_uid(self, series_instance_uid: str):
        """
        Return the series instance by the key "SeriesInstanceUID".

        :param series_instance_uid: series unique id for DICOM files
        :return: data frame of the series having the SeriesInstanceUID
        """

        df_series = self._df_series  # train & label
        df_series = df_series[df_series['SeriesInstanceUID'] == series_instance_uid]
        return df_series

    def filter_series_by_labels(self, labels: List[str], pos_labels: List[str], series_instance_uids: List[str]):
        """
        Filter series by labels and list of SeriesInstanceUIDs.
        After filtering, convert the labels to positive (1) and negative (0) labels.

        :param labels: list of all labels
        :param pos_labels: the labels to be defined as positive
        :param series_instance_uids: list of series unique ids for DICOM files
        :return: filtered data frame of series
        """

        # Filter by labels and list of SeriesInstanceUIDs
        df_series = self.filter_series(dataset='train',
                                       labels=labels,
                                       series_instance_uids=series_instance_uids)

        # Take the pairs of (SeriesInstanceUID, Class)
        df_series = df_series[['SeriesInstanceUID', 'Class']]
        if labels is not None:
            labels = set(labels)
            df_series = df_series[df_series['Class'].isin(labels)]

        # Sort series by SeriesInstanceUID
        df_series = df_series.sort_values(by='SeriesInstanceUID')

        # Convert string labels to numeric binary label 0/1
        neg_labels = list(set(labels) - set(pos_labels))
        df_series['Class'] = df_series['Class'].apply(self._convert_binary_class,
                                                      positive=pos_labels,
                                                      negative=neg_labels)
        return df_series

    def get_boxes_param(self, series_instance_uid: str):
        """
        Get the box information for the given series.

        :param series_instance_uid: series unique id for DICOM files
        :return: dictionary of box information of the input series
        """

        # Read the train dataset with boxes
        df_boxes = self._df_boxes

        # Get the series with SeriesInstanceUID
        series = self._df_series[self._df_series['SeriesInstanceUID'] == series_instance_uid]

        # If the series was not included in public dataset, return None
        if len(series) == 0:
            return None

        series = series.iloc[0].to_dict()

        df_boxes = df_boxes[df_boxes['PatientID'] == series['PatientID']]
        df_boxes = df_boxes[df_boxes['StudyUID'] == series['StudyUID']]
        df_boxes = df_boxes[df_boxes['View'] == series['View']]

        # If the series did not have box information in the public dataset, return None
        if len(df_boxes) == 0:
            return None

        # Take the box parameters
        df_boxes = df_boxes.loc[:,  self._boxes_key]

        return df_boxes.to_dict('records')

    def get_series_df(self):
        """
        Get the data frame of series from the dataset.

        :return: pandas series
        """
        return self._df_series

    def get_dicom_filepath(self, dicom_series: Dict[str, Any]):
        """
        Get the DICOM filepath: input/{series_instance_uid}.dcm

        When you download the DICOM files using the NBIA Data Retriever, there are two options for selecting
        the Directory Type, that correspond to those two paths in the .csv file: “Descriptive Directory Name” and
        “Classic Directory Name”. We are retrieving the SeriesInstanceUID from the classic path.

        :param dicom_series: dictionary of series
        :return: DICOM filepath with series instance uid
        """

        # Classic path: Breast-Cancer-Screening-DBT/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>
        series_instance_uid = self._extract_series_instance_uid(str(dicom_series["classic_path"]))

        # DICOM filepath with SeriesInstanceUID
        dicom_filepath = os.path.join(self._data_dir, f"{series_instance_uid}.dcm")
        return dicom_filepath

    def download_dicom(self, dicom_series: Dict[str, Any], current_number: int = 1, total_number: int = 1, unzip: bool = True):
        """
        Download the DICOM file by using its SeriesInstanceUID, which is retrived from its classic path.
        Save the DICOM file with its SeriesInstanceUID inside the data folder.

        :param dicom_series: dictionary of series including its classic path
        :param current_number: current number of series to download
        :param total_number: total number of series to download
        :param unzip: unzip the DICOM file after downloading or not
        :return: DICOM filepath where the file was saved
        """

        # Take IDs from the classic path
        classic_path = str(dicom_series["classic_path"]).split('/')
        _, patient_id, study_instance_uid, series_instance_uid = classic_path[0:4]

        # DICOM filepath with SeriesInstanceUID
        dicom_filepath_old = os.path.join(self._data_dir, "00000001.dcm")
        dicom_filepath = os.path.join(self._data_dir, f"{series_instance_uid}.dcm")

        if os.path.exists(dicom_filepath):
            print ("This file was already downloaded: {}/{}/{}".format(patient_id, study_instance_uid, series_instance_uid))
            return dicom_filepath

        print("Downloading {}... ({}/{})".format(f"{patient_id}/{study_instance_uid}/{series_instance_uid}", current_number, total_number))

        # SeriesInstanceUID parameter is required for this API call.
        r_data = requests.get(
            f"{self._base_url}/getImage",
            params={"SeriesInstanceUID": series_instance_uid},
        )

        if not r_data.ok:
            print("Got response status {}, skipping {}".format(r_data.status_code,
                  f"{patient_id}/{study_instance_uid}/{series_instance_uid}"))

        if unzip:
            file = ZipFile(io.BytesIO(r_data.content))
            file.extractall(path=self._data_dir)
            os.rename(dicom_filepath_old, dicom_filepath)
        else:
            with open(self._data_dir.with_suffix(".zip"), "wb") as zip_file:
                zip_file.write(r_data.content)

        return dicom_filepath

    def download_all_dicom(self, series_list: List[Dict[str, Any]], unzip: bool = True):
        """
        Download the DICOM files by using their SeriesInstanceUID, which are retrived from their classic paths.
        Save them with its SeriesInstanceUIDs inside the data folder.

        :param series_list: list of dictionaries of series, including classic paths
        :param unzip: unzip all the DICOM file after downloading or not
        :return: list of DICOM filepaths
        """
        dicom_filepaths = []

        for i, series in enumerate(series_list, start=1):
            dicom_filepath = self.download_dicom(dicom_series=series,
                                                 current_number=i,
                                                 total_number=len(series_list),
                                                 unzip=unzip)
            dicom_filepaths.append(dicom_filepath)

        return dicom_filepaths

    def download_all_series(self, destination_path: str):
        """
        Download DBT series from the cloud database with the 'Modality' of 'MG', i.e. mammography
        Save the DBT series in a CSV file.
        This method is not necessary for our project.

        :param destination_path: path to save the CSV file
        :return: downloaded series
        """

        if destination_path is not None:
            if os.path.exists(destination_path):
                print("Metadata of series was already downloaded as a CSV file in {}.".format(destination_path))

                # Read the CSV file and convert it to a list of dictionaries
                print("Reading the CSV file...")
                df = pd.read_csv(destination_path)
                series = df.to_dict('records')

                return series

        print("Downloading series metadata...")

        # Make a GET request to retrieve a series
        # The response from the API call is stored in the `series` variable as a list of dictionaries
        # Each dictionary represents a series with its attributes e.g., SeriesInstanceUID, StudyInstanceUID, Modality etc.
        series: List[Dict[str, Any]] = requests.get(
            f"{self._base_url}/getSeries",  # API endpoint to retrieve series
            params={"Collection": "Breast-Cancer-Screening-DBT"},  # Query parameter to specify the collection
            timeout=None  # Timeout duration for the request (None means no timeout)
        ).json()

        if destination_path is not None:
            print("Saving metadata of series as a CSV file in {}.".format(destination_path))

            df = pd.DataFrame(series)
            df.to_csv(destination_path, index=False)

        return series

    @staticmethod
    def _extract_series_instance_uid(classic_path: str):
        classic_path_ = classic_path.split('/')
        series_instance_uid = classic_path_[3]
        return series_instance_uid

    @staticmethod
    def _convert_binary_class(label: str, positive: List[str], negative: List[str]):
        if label in positive:
            return 1
        elif label in negative:
            return 0

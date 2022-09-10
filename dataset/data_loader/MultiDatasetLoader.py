"""The dataloader for PURE datasets.

Details for the PURE Dataset see https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
If you use this dataset, please cite the following publication:
Stricker, R., Müller, S., Gross, H.-M.
Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
"""
import os
import cv2
import glob
import json
import numpy as np
import re
from dataset.data_loader.BaseLoader import BaseLoader
from utils.utils import sample
from multiprocessing import Pool, Process, Value, Array, Manager
from tqdm import tqdm


class MultiDatasetLoader(BaseLoader):
    """The data loader for the PURE dataset."""


    def __init__(self, name, data_path, config_data):
        """Initializes an PURE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 01-01/
                     |      |-- 01-01/
                     |      |-- 01-01.json
                     |   |-- 01-02/
                     |      |-- 01-02/
                     |      |-- 01-02.json
                     |...
                     |   |-- ii-jj/
                     |      |-- ii-jj/
                     |      |-- ii-jj.json
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)


    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):

        print('Preprocess currently unsupported For MultiDatasetLoader...')
        print('Please update in config file.')
        print('Currently data can only be passed through csv option')
        print('If preprocess is required please run the preprocess script.')
        raise ValueError(self.name, 'Preprocess Unsupported For MultiDatasetLoader')

        return None

    
    def load(self):
        """Loads the preprocessing data."""

        file_list = self.file_list # get list of files in 


        inputs = glob.glob(os.path.join(self.cached_path, "*input*.npy"))
        if inputs == []:
            raise ValueError(self.name+' dataset loading data error!')
        labels = [input.replace("input", "label") for input in inputs]
        assert (len(inputs) == len(labels))
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)
        print("loaded data len:",self.len)


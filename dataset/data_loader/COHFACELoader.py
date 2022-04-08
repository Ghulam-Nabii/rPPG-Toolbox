"""The dataloader for COHFACE datasets.

Details for the COHFACE Dataset see https://www.idiap.ch/en/dataset/cohface
If you use this dataset, please cite the following publication:
Guillaume Heusch, André Anjos, Sébastien Marcel, “A reproducible study on remote heart rate measurement”, arXiv, 2016.
http://publications.idiap.ch/index.php/publications/show/3688
"""
import os
import cv2
import numpy as np
import h5py
from dataset.data_loader.BaseLoader import BaseLoader
from utils.utils import sample


class COHFACELoader(BaseLoader):
    """The data loader for the COHFACE dataset."""

    def __init__(self, name, data_dirs, config_data):
        """Initializes an COHFACE dataloader.
            Args:
                data_dirs(list): A list of paths storing raw video and bvp data.
                Each contains 4 one-minute videos for one subject.
                e.g. [RawData/1,RawData/2,...,RawData/n] for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 1/
                     |      |-- 0/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |      |...
                     |      |-- 3/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |...
                     |   |-- n/
                     |      |-- 0/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |      |...
                     |      |-- 3/
                     |          |-- data.avi
                     |          |-- data.hdf5
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_dirs, config_data)

    def preprocess_dataset(self, config_preprocess):
        """Preprocesses the raw data."""
        file_num = len(self.data_dirs)
        for i in range(file_num):
            frames = self.read_video(
                os.path.join(
                    self.data_dirs[i]["path"],
                    "data.avi"))
            bvps = self.read_wave(
                os.path.join(
                    self.data_dirs[i]["path"],
                    "data.hdf5"))
            bvps = sample(bvps, frames.shape[0])
            frames_clips, bvps_clips = self.preprocess(
                frames, bvps, config_preprocess, False)
            self.len += self.save(frames_clips, bvps_clips,
                                  self.data_dirs[i]["index"])

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()

        frames = list()

        # cv2.imwrite("temp/exemple.png", frame)
        while(success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        f = h5py.File(bvp_file, 'r')
        pulse = f["pulse"][:]
        return pulse

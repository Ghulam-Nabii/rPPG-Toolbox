"""The dataloader for UBFC datasets.

Details for the UBFC-RPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import os
import cv2
import numpy as np
from dataset.data_loader.data_loader import data_loader


class UBFC_loader(data_loader):
    """The data loader for the UBFC dataset."""

    def __init__(self, name,data_dirs,config_data):
        """Initializes an UBFC dataloader.
            Args:
                data_dirs(list): A list of paths storing raw video and bvp data.
                e.g. [UBFC/subject1,UBFC/subject2,...,UBFC/subjectn] for below dataset structure:
                -----------------
                     UBFC/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name,data_dirs,config_data)

    def __len__(self):
        """Returns the length of the dataset."""
        return self.len

    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        input = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        input = np.transpose(input, (3, 0, 1, 2))
        return input, label

    def preprocess_dataset(self, config_preprocess):
        """Preprocesses the raw data."""
        file_num = len(self.data_dirs)
        for i in range(file_num):
            frames = self.read_video(
                os.path.join(
                    self.data_dirs[i],
                    "vid.avi"))
            bvps = self.read_wave(
                os.path.join(
                    self.data_dirs[i],
                    "ground_truth.txt"))
            frames_clips,bvps_clips = self.preprocess(frames,bvps,config_preprocess,False)
            self.len += self.save(frames_clips, bvps_clips, self.data_dirs[i])
        print(self.len)


    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while(success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)

"""The dataloader for COHFACE datasets.

Details for the COHFACE Dataset see https://www.idiap.ch/en/dataset/cohface
If you use this dataset, please cite the following publication:
Guillaume Heusch, André Anjos, Sébastien Marcel, “A reproducible study on remote heart rate measurement”, arXiv, 2016.
http://publications.idiap.ch/index.php/publications/show/3688
"""
import os
import cv2
import glob
import numpy as np
import h5py
import re
from dataset.data_loader.BaseLoader import BaseLoader
from utils.utils import sample


class COHFACELoader(BaseLoader):
    """The data loader for the COHFACE dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an COHFACE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
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
        super().__init__(name, data_path, config_data)

    def get_data(self, data_path):
        """Returns data directories under the path(For COHFACE dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*")
        if (data_dirs == []):
            raise ValueError(self.name+ " dataset get data error!")
        dirs = list()
        for data_dir in data_dirs:
            for i in range(4):
                subject = os.path.split(data_dir)[-1]
                dirs.append({"index": int('{0}0{1}'.format(subject, i)), "path": os.path.join(data_dir, str(i))})
        return dirs

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):

        print('Data Dirs: Girish Test')
        print(data_dirs)
        raise ValueError(self.name+ " FORCE QUIT GIRISH")


        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        print("file_num:",file_num)

        if (file_num != 160):
            raise ValueError(self.name+ " Incorrect num of videos. COHFACE dataset should have 160 videos.")


        choose_range = range(0,file_num)
        if (begin !=0 or end !=1):
            # divide file num by 4. Each subj has 4 vids, and we do not want subj overlap between train/val/test
            choose_range = range(4 * int(begin * file_num/4), int(end * file_num/4)) 
            print(choose_range)

        pbar = tqdm(list(choose_range))
        # multi_process
        p_list = []
        running_num = 0
        for i in choose_range:
            process_flag = True
            while (process_flag):         # ensure that every i creates a process
                if running_num < 32:       # in case of too many processes
                    p = Process(target=self.preprocess_dataset_subprocess, args=(data_dirs,config_preprocess,i))
                    p.start()
                    p_list.append(p)
                    running_num +=1
                    process_flag = False
                for p_ in p_list:
                    if (not p_.is_alive() ):
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()
        # append all data path and update the length of data
        inputs = glob.glob(os.path.join(self.cached_path, "*input*.npy"))
        if inputs == []:
            raise ValueError(self.name + ' dataset loading data error!')
        labels = [input.replace("input", "label") for input in inputs]
        assert (len(inputs) == len(labels))
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)

        # OLD LOADER
        file_num = len(data_dirs)
        for i in range(file_num):
            frames = self.read_video(
                os.path.join(
                    data_dirs[i]["path"],
                    "data.avi"))
            bvps = self.read_wave(
                os.path.join(
                    data_dirs[i]["path"],
                    "data.hdf5"))
            bvps = sample(bvps, frames.shape[0])
            frames_clips, bvps_clips = self.preprocess(
                frames, bvps, config_preprocess, False)
            self.len += self.save(frames_clips, bvps_clips,
                                  data_dirs[i]["index"])

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

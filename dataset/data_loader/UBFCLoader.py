"""The dataloader for UBFC datasets.

Details for the UBFC-RPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import os
import cv2
import glob
import numpy as np
import re
from dataset.data_loader.BaseLoader import BaseLoader


class UBFCLoader(BaseLoader):
    """The data loader for the UBFC dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an UBFC dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
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
        super().__init__(name, data_path, config_data)

    def get_data(self, data_path):
        """Returns data directories under the path(For UBFC dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if (data_dirs == []):
            raise ValueError(self.name+ " dataset get data error!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, inputs, labels, len_num):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']
        frames = self.read_video(
            os.path.join(
                data_dirs[i]['path'],
                filename, ""))
        bvps = self.read_wave(
            os.path.join(
                data_dirs[i]['path'],
                "{0}.json".format(filename)))
        bvps = sample(bvps, frames.shape[0])
        frames_clips, bvps_clips = self.preprocess(
            frames, bvps, config_preprocess, config_preprocess.LARGE_FACE_BOX)

        count,input_name_list, label_name_list= self.save_multi_process(frames_clips, bvps_clips,
                              saved_filename)
        inputs[i] = input_name_list
        labels[i] = label_name_list
        len_num.value = len_num.value + count



    def preprocess_dataset(self, data_dirs, config_preprocess,begin, end):
        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        print("file_num:",file_num)
        choose_range = range(0,file_num)
        if (begin !=0 or end !=1):
            choose_range = range(int(begin*file_num), int(end * file_num))
            print(choose_range)
        # multi_process
        p_list = []
        with Manager() as manager:
            inputs_share = manager.dict()
            labels_share = manager.dict()
            len_num = Value('i', 0)
            for i in choose_range:
                p = Process(target=self.preprocess_dataset_subprocess,
                            args=(data_dirs, config_preprocess, i, inputs_share, labels_share, len_num))
                p.start()
                p_list.append(p)
            # join all processes
            for p_ in p_list:
                p_.join()
            # append all data path and update the length of data
            for index in choose_range:
                for input in inputs_share[index]:
                    self.inputs.append(input)
                for label in labels_share[index]:
                    self.labels.append(label)
            self.len = len_num.value


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

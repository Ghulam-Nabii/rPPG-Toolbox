""" The dataloader for Tang datasets.

"""

import os
import cv2
import glob
import numpy as np
import re
from BaseLoader import BaseLoader
from multiprocessing import Pool, Process, Value, Array, Manager
from tqdm import tqdm
import pandas as pd
import skvideo.io

class TANGLoader(BaseLoader):
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
        print(data_path)
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if (data_dirs == []):
            raise ValueError(self.name + " dataset get data error!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']
        frames = self.read_video(
                os.path.join(data_dirs[i]['path'],"10_0_cut.mp4"))
        bvps = self.read_wave(
                os.path.join(data_dirs[i]['path'],"10.xlsx"),
                frames_length=len(frames))
        frames_clips, bvps_clips = self.preprocess(
            frames, bvps, config_preprocess, config_preprocess.LARGE_FACE_BOX)

        count, input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips,
                                                                          saved_filename)

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        print("file_num:", file_num)
        choose_range = range(0, file_num)
        if (begin != 0 or end != 1):
            choose_range = range(int(begin * file_num), int(end * file_num))
            print(choose_range)
        pbar = tqdm(list(choose_range))
        # multi_process
        p_list = []
        running_num = 0
        for i in choose_range:
            process_flag = True
            while (process_flag):
                if running_num < 32:
                    p = Process(target=self.preprocess_dataset_subprocess,
                                args=(data_dirs, config_preprocess, i))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if (not p_.is_alive()):
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

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        # RGB = skvideo.io.vread(video_file)
        # print(RGB.shape)
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(frameCount,frameWidth,frameHeight)
        frames = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        frames_count = 0
        success = True
        while (frames_count < frameCount and success):
            success, frames[frames_count] = cap.read()
            frames_count += 1
        print("frames",frames.shape)
        # np.save('/data1/acsp/Yuzhe_Zhang/rPPG-Toolbox/PreprocessedData/1/output', frames)
        return frames

    @staticmethod
    def read_wave(bvp_file, frames_length):
        """Reads a bvp signal file."""
        gt_pulse, gt_time = generate_pulse_gt(bvp_file, 0)
        gt_pulse = np.array(gt_pulse)
        # print("gt_pulse_num",len(gt_pulse))
        # print("gt_time_num",len(gt_time))
        # print(frames_length)
        gt_start_time = gt_time[0]
        gt_end_time = gt_start_time + frames_length/60 * 1000
        for i in range(len(gt_pulse)):
            if gt_time[i] >= gt_end_time:
                print(i, gt_time[i])
                gt_pulse = gt_pulse[0:i]
                gt_time = gt_time[0:i]
                break
        # print("gt_start_time",gt_start_time)
        # print("gt_end_time",gt_end_time)
        re_time = np.linspace(gt_start_time, gt_end_time, frames_length)
        re_gt_pulse = np.interp(re_time, gt_time, gt_pulse)
        gt_length = len(re_gt_pulse)
        # print("gt_length:", gt_length)
        return re_gt_pulse


def generate_pulse_gt(file_path, experiment_id):
    gt_time = []  # an empty list to store the first column
    gt_pulse = []  # an empty list to store the second column
    call_in = False
    with open(file_path, 'r') as rf:
        # reader = csv.reader(rf, delimiter=',')
        df = pd.read_excel(file_path, sheet_name=str(experiment_id), header=None)
        df[1] = df[1].apply(int, base=16)
        gt_pulse = list(df[1])
        gt_time = list(df[0])
    return gt_pulse, gt_time
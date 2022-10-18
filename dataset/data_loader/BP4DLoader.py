"""The dataloader for BP4D datasets.
"""
import glob
import glob
import json
import os
import re
import mat73

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

class BP4DLoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an PURE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:

        """
        super().__init__(name, data_path, config_data)

    def get_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*.mat")
        if not data_dirs:
            raise ValueError(self.name + " dataset get data error!")
        dirs = list()
        for data_dir in data_dirs:
            subject_data = os.path.split(data_dir)[-1].replace('.mat', '')
            subj_sex = subject_data[0]
            subject = int(subject_data[1:4])
            index = subject_data
            dirs.append({"index": subject_data, "path": data_dir, "subject": subject, "sex": subj_sex})
        return dirs

    def get_data_subset(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # get info about the dataset: subject list and num vids per subject
        m_data_info = dict() # data dict for Male subjects
        f_data_info = dict() # data dict for Female subjects
        for data in data_dirs:

            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            sex = data['sex']

            trials_to_skip = ['F041T7', 'F054T9'] # GIRISH TO DO, Talk to BP4D ppl about this
            if index in trials_to_skip:
                continue

            if sex == 'M':
                # creates a dictionary of data_dirs indexed by subject number
                if subject not in m_data_info:  # if subject not in the data info dictionary
                    m_data_info[subject] = []  # make an emplty list for that subject
                # append a tuple of the filename, subject num, trial num, and chunk num
                m_data_info[subject].append({"index": index, "path": data_dir, "subject": subject, "sex": sex})


            elif sex == 'F':
                # creates a dictionary of data_dirs indexed by subject number
                if subject not in f_data_info:  # if subject not in the data info dictionary
                    f_data_info[subject] = []  # make an emplty list for that subject
                # append a tuple of the filename, subject num, trial num, and chunk num
                f_data_info[subject].append({"index": index, "path": data_dir, "subject": subject, "sex": sex})

        # List of Male subjects
        m_subj_list = list(m_data_info.keys())  # all subjects by number ID
        m_subj_list.sort()
        m_num_subjs = len(m_subj_list)  # number of unique subjects

        # get male split of data set (depending on start / end)
        m_subj_range = list(range(0, m_num_subjs))
        if begin != 0 or end != 1:
            m_subj_range = list(range(int(begin * m_num_subjs), int(end * m_num_subjs)))
        print('Used Male subject ids for split:', [m_subj_list[i] for i in m_subj_range])

        # List of Female subjects
        f_subj_list = list(f_data_info.keys())  # all subjects by number ID 
        f_subj_list.sort()
        f_num_subjs = len(f_subj_list)  # number of unique subjects

        # get female split of data set (depending on start / end)
        f_subj_range = list(range(0, f_num_subjs))
        if begin != 0 or end != 1:
            f_subj_range = list(range(int(begin * f_num_subjs), int(end * f_num_subjs)))
        print('Used Female subject ids for split:', [f_subj_list[i] for i in f_subj_range])

        # compile file list
        file_info_list = []

        # add male subjects to file list
        for i in m_subj_range:
            subj_num = m_subj_list[i]
            subj_files = m_data_info[subj_num]
            file_info_list += subj_files  # add file info to file_list (tuple of fname, subj ID, trial num, # chunk num)

        # add female subjects to file list
        for i in f_subj_range:
            subj_num = f_subj_list[i]
            subj_files = f_data_info[subj_num]
            file_info_list += subj_files  # add file info to file_list (tuple of fname, subj ID, trial num, # chunk num)

        return file_info_list

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = data_dirs[i]['path']
        saved_filename = data_dirs[i]['index']
        
        frames = self.read_video(filename)
        labels = self.read_labels(os.path.join(filename))

        # GIRISH TO DO
        if frames.shape[0] != labels.shape[0]:  # CHECK IF ALL DATA THE SAME LENGTH
            raise ValueError(self.name, 'frame and label time axis not the same')

        frames_clips, labels_clips = self.preprocess(frames, labels, config_preprocess, config_preprocess.LARGE_FACE_BOX)
        count, input_name_list, label_name_list = self.save_multi_process(frames_clips, labels_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def preprocess(self, frames, labels, config_preprocess, large_box=False):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Bvp signal labels for a video.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            large_box(bool): Whether to use a large bounding box in face cropping, e.g. in moving situations.
        """
        # frames = self.resize(
        #     frames,
        #     config_preprocess.DYNAMIC_DETECTION,
        #     config_preprocess.DYNAMIC_DETECTION_FREQUENCY,
        #     config_preprocess.W,
        #     config_preprocess.H,
        #     config_preprocess.LARGE_FACE_BOX,
        #     config_preprocess.CROP_FACE,
        #     config_preprocess.LARGE_BOX_COEF)
        
        # data_type
        data = list()
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c[:-1, :, :, :])
            elif data_type == "Normalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c)[:-1, :, :, :])
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=3)

        # TO DO: GIRISH - is the below thinking correct
        # normalize both time-series, periodic values (bp and resp waves), all other values can be left as is
        bp_wave = labels[:, 0]
        resp_wave = labels[:, 5]
        labels = labels[:-1] # adjust size to match normalized size

        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "Normalized":
            bp_wave = BaseLoader.diff_normalize_label(bp_wave)
            resp_wave = BaseLoader.diff_normalize_label(resp_wave)
            labels[:, 0] = bp_wave
            labels[:, 5] = resp_wave
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bp_wave = BaseLoader.standardized_label(bp_wave)[:-1]
            resp_wave = BaseLoader.standardized_label(resp_wave)[:-1]
            labels[:, 0] = bp_wave
            labels[:, 5] = resp_wave
        
        # Chunk clips and labels
        if config_preprocess.DO_CHUNK:
            frames_clips, labels_clips = self.chunk(data, labels, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            labels_clips = np.array([labels])

        return frames_clips, labels_clips

    def chunk(self, frames, labels, chunk_length):
        """Chunks the data into clips."""
        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        labels_clips = [labels[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(labels_clips)

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        print("file_num:", file_num)
        choose_range = range(0, file_num)

        if begin != 0 or end != 1:
            data_dirs = self.get_data_subset(data_dirs, begin, end)
            choose_range = range(0, len(data_dirs))
        print(choose_range)

        file_list_dict = self.multi_process_manager(data_dirs, config_preprocess, choose_range)
        self.build_file_list(file_list_dict, len(list(choose_range))) # build file list
        self.load() # load all data and corresponding labels (sorted for consistency)

    @staticmethod
    def read_video(file_path):
        """Reads a video file, returns frames(T,H,W,3) """
        f = mat73.loadmat(file_path)
        frames = f['X']
        return np.asarray(frames)

    @staticmethod
    def read_labels(file_path):
        """Reads a bvp signal file."""
        f = mat73.loadmat(file_path)
        keys = list(f.keys())
        data_len = f['X'].shape[0]
        keys.remove('X')

        # GIRISH TO DO: Eventually make ALL
        #labels = np.zeros((data_len, len(keys)))
        labels = np.zeros((data_len, 8))
        # labels by index in array:
        # 0: bp_wave, 1: hr_bpm, 2: systolic_bp, 3: diastolic_bp, 4: mean_bp, 5: resp_wave, 6: resp_bpm, 7: eda, [8,47]: AUs
        labels_order_list = ['bp_wave', 'HR_bpm', 'systolic_bp', 'diastolic_bp', 'mean_bp', 'resp_wave', 'resp_bpm', 'eda']
        # for l in labels_order_list: 
        #     keys.remove(l)
        # labels_order_list += keys 

        # re-order labels numpy mtx
        for i in range(len(labels_order_list)):
            labels[:, i] = f[labels_order_list[i]]

        return np.asarray(labels)

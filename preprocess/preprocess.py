import os
import cv2
import csv
import pandas as pd

def generate_pulse_gt(file_path,experiment_id):
    gt_time = []          # an empty list to store the first column
    gt_pulse = []         # an empty list to store the second column
    call_in = False
    with open(file_path, 'r') as rf:
        # reader = csv.reader(rf, delimiter=',')
        df = pd.read_excel(file_path,sheet_name=str(experiment_id),header=None)
        df[1] = df[1].apply(int, base=16)
        gt_pulse = list(df[1])
        gt_time = list(df[0])       
    return gt_pulse, gt_time

def hex2dec(hex):
    """Convert a hexadecimal string to a decimal number"""
    hex = "0x" + hex
    result_dec = int(hex, 0)
    return result_dec

def YUV420p2yuvfile(filepath, fname, YUVname, bitdepth = 8):
    """
    Convert YUV 420p video to raw YUV file.
     
    input: 
    
    - inputpath: dir of the video file (Color Space: YUV 420p). ending in '/'
    - fname: filename, w/ suffix.
    - YUVname:  YUV file name. Due to YUV file siez restriction, each file should be less than 4GB. Thus, YUV data would 
    be divided to 15s chunks due to the duration of the video.
    - bitdepth: 8bit / 10bit
    """
    if bitdepth == 10:
        os.system("cd " + filepath)
        os.system("ffmpeg -i " + filepath + fname + " -pix_fmt yuv420p10le -to 00:00:15 " + filepath + YUVname[:-4] + "_0.yuv")
        os.system("ffmpeg -i " + filepath + fname + " -pix_fmt yuv420p10le -ss 00:00:15 " + filepath + YUVname[:-4] + "_1.yuv")
    elif bitdepth == 8:
        os.system("cd " + filepath)
        os.system("ffmpeg -i " + filepath + fname + " -pix_fmt yuv420p " + filepath + YUVname[:-4] + ".yuv")
        # os.system("ffmpeg -i " + filepath + fname + " -pix_fmt yuv420p -ss 00:00:15 " + filepath + YUVname[:-4] + "_1.yuv")

if __name__ == "__main__":
    filepath = "/Volumes/Dirk_SSD/Sinan_10bit_1hz_Check_2022_4_27/"

    for file in os.listdir(filepath):
        if not "_HDR_"  in file and not file.endswith(".csv") and not file.startswith("._"):
            print("processing: ", file)
            YUVname = file[:-4] + ".yuv"
            print(YUVname)
            YUV420p2yuvfile(filepath, file, YUVname, 8)

# HDR10_2_YUV(filepath, fname, YUVname)
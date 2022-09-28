import numpy as np
import os
import cv2
from pyrsistent import inc
import skvideo.io
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
import scipy.signal as signal
import math
import pathlib
import glob
import time
from skimage.util import img_as_float
#----------------------------------------------------------------
# video related
#----------------------------------------------------------------

def preprocess_raw_video(videoFilePath,start_time, dim1=320,dim2=240,normalize = False):


    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath);
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
    Xsub = np.zeros((totalFrames, dim1, dim2, 3),dtype = np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    dims = img.shape
    print(dims)
    print("Orignal Height", height)
    print("Original width", width)
    #########################################################################
    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
        # vidLxL = cv2.resize(img_as_float(img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim1, dim2), interpolation = cv2.INTER_AREA)
        vidLxL = cv2.resize(img_as_float(img), (dim1, dim2), interpolation = cv2.INTER_AREA)
        vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
        vidLxL = cv2.cvtColor(vidLxL.astype(np.float32), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1/255)] = 1/255
        Xsub[i, :, :, :] = vidLxL
        success, img = vidObj.read() # read the next one
        i = i + 1
    
    plt.imshow(Xsub[0])
    plt.title('Sample Preprocessed Frame')
    plt.show()
    #########################################################################
    # Normalized Frames in the motion branch
    if normalize == True:
        print('Normalized Frames in the motion branch')
        normalized_len = len(t) - 1
        dXsub = np.zeros((normalized_len, dim1, dim2, 3), dtype = np.float32)
        for j in range(normalized_len - 1):
            dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
        dXsub = dXsub / np.std(dXsub)
        #########################################################################
        # Normalize raw frames in the apperance branch
        Xsub = Xsub - np.mean(Xsub)
        Xsub = Xsub  / np.std(Xsub)
        Xsub = Xsub[:totalFrames-1, :, :, :]
        #########################################################################
        # Plot an example of data after preprocess
        dXsub = np.concatenate((dXsub, Xsub), axis = 3)
    print(Xsub.shape)
    # print(Xsub)
    RGB_length,full_video_duration =video_duration(videoFilePath)
    # print(RGB_length,full_video_duration)
    end_time = start_time+60
    start_frame = int(start_time*RGB_length/full_video_duration)
    end_frame = int(end_time*RGB_length/full_video_duration)
    frames_num =end_frame-start_frame
    if frames_num == 1800:
        Xsub = Xsub[start_frame:end_frame]
    elif frames_num<1800:
        Xsub = Xsub[start_frame:end_frame]
        plusXsub = np.zeros((1800-frames_num, dim1, dim2, 3),dtype = np.float32)
        print(frames_num,len(plusXsub))
        Xsub = np.concatenate((Xsub,plusXsub ), axis = 0)
        print(Xsub[-5:])
    else :
        Xsub = Xsub[start_frame:start_frame+1800]
    Xsub = Xsub.astype(np.float32)
    print('after process: ', Xsub.shape)
    return Xsub,frames_num

#----------------------------------------------------------------
# OS related
#----------------------------------------------------------------

def recursive_listdir():
    """
    return all the files with related suffix in all subdirs. (e.g. ".MOV")
    just for show
    """
    filepath = ""
    for f in glob.glob(filepath + "**/*.MOV", recursive=True):
        print(f)
        print("mat name", str.replace(str.split(f, "/")[-2] + "_" + str.split(f, "/")[-1][:-4] + ".mat", " ", ""))
    
#----------------------------------------------------------------
# str operation
#----------------------------------------------------------------
def str_replace():
    str.replace("Hello world", " ", "")

def str_split():
    str.split("a/b", "/")[0] 

# ----------------------------------------------------------------
# Visualization
#----------------------------------------------------------------
def makeTable(labelrow, labelcol, Texts, title):
    """
    make a h * w table;
    labelrow: [h,]
    labelcol: [w,]
    Texts: [h,w]
    """
    
    fig, ax = plt.subplots(2,1) 
    ax[0].set_axis_off() 
    table = ax[0].table( 
    cellText = Texts,  
    rowLabels = labelrow,  
    colLabels = labelcol, 
    rowColours =["palegreen"] * len(labelcol),  
    colColours =["palegreen"] * len(labelrow), 
    cellLoc ='center',  
    loc ='upper left')    
    
    ax[1].set_axis_off()
    table2 = ax[1].table( 
    cellText = Texts,  
    rowLabels = labelrow,  
    colLabels = labelcol, 
    rowColours =["palegreen"] * len(labelcol),  
    colColours =["palegreen"] * len(labelrow), 
    cellLoc ='center',  
    loc ='upper left')   
   
    ax[0].set_title(title, 
             fontweight ="bold") 
    ax[1].set_title(title, 
             fontweight ="bold") 
    plt.show() 

#----------------------------------------------------------------
# math
#----------------------------------------------------------------
def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

#----------------------------------------------------------------
# PPG
#----------------------------------------------------------------
def calculate_HR(fs, signal):
    N = 30 * fs
    pulse_fft = np.expand_dims(signal, 0)
    f, pxx = scipy.signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
    fmask = np.argwhere((f >= 0.75) & (f <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
    # fmask = np.argwhere(((f >= 0.75) & (f <= 0.95)) | ((f > 1.05) & (f <= 2.5)))
    frange = np.take(f, fmask)
    HR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60
    return HR, np.take(f, fmask), np.take(pxx, fmask)

def bvpsnr(BVP, FS, HR, PlotTF):
    '''
    :param BVP:
    :param FS:
    :param HR:
    :param PlotTF:
    :return:
    '''
    HR_F = HR / 60
    NyquistF = FS / 2
    FResBPM = 0.5
    N = (60 * 2 * NyquistF) / FResBPM

    ##Construct Periodogram
    F, Pxx = signal.periodogram(BVP, FS, nfft=N, window="hamming")
    GTMask1 = (F >= HR_F - 0.1) & (F <= HR_F + 0.1)
    GTMask2 = (F >= (HR_F * 2 - 0.2)) & (F <= (HR_F * 2 + 0.2))
    temp = GTMask1 | GTMask2
    SPower = np.sum(Pxx[temp])
    FMask2 = (F >= 0.5) & (F <= 4)
    AllPower = np.sum(Pxx[FMask2])
    SNR = 10 * math.log10(SPower / (AllPower - SPower))
    print("SignalNoiseRatio", SNR)
    return SNR

#----------------------------------------------------------------
# I/O
#----------------------------------------------------------------
def Read_Writemat():
    """
    not a fx actually. Only a quicknote for writing mats
    """
    sio.savemat("a.mat", {"dXsub":"nparray", "dysub" : "nparray", "dbsub":"nparray"})
    mat = sio.loadmat('file.mat')

#----------------------------------------------------------------
# CV2 Manipulation
#----------------------------------------------------------------
def selectROI(path, wname, mode = 'video', increase_bright=True):
    """
    get the selected [x,y,h,w] parameter out of a vid's first frame.
    
    path: path of the image/video
    wname: cv2 window name
    mode: video, (to be added), image
    increase_bright: increase brightness of image for dim photo. Better ROI
    """

    if mode == "video":
        vidcapture = cv2.VideoCapture(path)
        # Wait for the video to stabilize
        success, img = vidcapture.read()
        for i in range(180):
            success, img = vidcapture.read()
            
        if increase_bright:
            img = increase_brightness(img)
            
        # cv2.namedWindow(wname,cv2.WINDOW_NORMAL)
        cv2.namedWindow(wname, cv2.WINDOW_AUTOSIZE   )
        r = cv2.selectROI(wname, img)
        print('Coordiantes: ', r)
        
        cv2.waitKey(0) # Press enter again to close both windows
        cv2.destroyWindow(wname)
            
        return r

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

def increase_brightness(img, value=30):
    """
    increase brightness for an given image.

    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


#----------------------------------------------------------------
# Matrix Manipulation
#----------------------------------------------------------------
def matrix_1x122x2(input_m):
    """
    12 -> 1122
    34    1122
          3344
          3344  
    input_m: [a, b]
    output_m: [2a, 2b]
    """
    input_m = np.array(input_m)
    b = input_m[:,:,np.newaxis]
    c = np.concatenate([b,b],axis=2)
    d = c[:,np.newaxis, :,:]
    e = np.concatenate([d,d],axis=1)
    f = e.reshape((input_m.shape[0] * 2, input_m.shape[1] * 2))
    
    return f

def matrix_row_col_add(rowarray, colarray):
    """
    e.g. 
    rowarray [1,2,3]
    colarray [4,7,9]
    output: [5,6,7]
            [8,9,10]
            [10,11,12]
    """
    # rowarray = np.array(rowarray)
    # colarray = np.array(colarray)
    row = rowarray[np.newaxis, :]
    col = colarray[:, np.newaxis]
    rows =  np.repeat(row, col.shape[0], axis=0)
    print("repeat 1 done!")
    print(rows.shape)
    cols = np.repeat(col, row.shape[1], axis=1)
    print("repeat 2 done!")
    print(cols.shape)
    np.save("rows.npy", rows)
    np.save("cols.npy", cols)
    mat = rows + cols
    print(mat.shape)
    np.save("mat.npy", mat)
    
    return mat
#----------------------------------------------------------------
# Color Conversion
#----------------------------------------------------------------

def RGB2YCbCr_ITU_BT709(R, G, B):
    # https://forum.blackmagicdesign.com/viewtopic.php?f=12&t=29413
    # Y: 16 - 25; Cb/Cr: 16-240, HDTV
    Y = 0.183 * R + 0.614 * G + 0.062 * B + 16
    Cb = -0.101 * R - 0.339 * G + 0.439 * B + 128
    Cr = 0.439 * R - 0.399 * G - 0.040 * B + 128
    return Y, Cb, Cr

def RGB2YCbCr_ITU_BT601(R, G, B):
    # https://forum.blackmagicdesign.com/viewtopic.php?f=12&t=29413
    # SDTV  Y: 16-240, Cb/ Cr: 16-235.
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128
    return Y, Cb, Cr

def YCbCr2RGB_ITU_BT601(Y, Cb, Cr):
    # https://forum.blackmagicdesign.com/viewtopic.php?f=12&t=29413
    # SDTV  Y: 16-240, Cb/ Cr: 16-235.
    R = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
    G = 1.164 * (Y - 16) - 0.813 * (Cr - 128) - 0.391 * (Cb - 128)
    B = 1.164 * (Y - 16) + 2.018 * (Cb - 128)
    return R, G, B

def YCbCr2RGB_ITU_BT709(Y, Cb, Cr):
    # https://forum.blackmagicdesign.com/viewtopic.php?f=12&t=29413
    # Y: 16 - 25; Cb/Cr: 16-240, HDTV
    R = 1.164 * (Y - 16) + 1.793 * (Cr - 128)
    G = 1.164 * (Y - 16) - 0.534 * (Cr - 128) - 0.213 * (Cb - 128)
    B = 1.164 * (Y - 16) + 2.115 * (Cb - 128)
    return R,G,B

def YUV420p2YUV(YUVarray, w, h, f_num, d_type = "uint16"):
    """
    extract Y,U,V from yuv file generated from ffmpeg.
    https://stackoverflow.com/questions/53467655/import-yuv-as-a-byte-array
    
    input:
    - YUVarray (lenghth,) byte array
    - w (————) of original video
    - h (|) of original video
    - fnum: frame num
    - d_type: uint8 or uint16. depends on the bit depth of the video.
    
    Output:
    - Y (framenum, h, w)
    - U (framenum, h//2, w//2)
    - V (framenum, h//2, w//2)
    """
    ##TODO-------- NEEDS REVISION!!!!!!!!!!!!!!!!!
    YUVarray = np.array(YUVarray)
    px = w * h
    f_num = int(f_num)
    print(f_num, h, w)
    print("hey 1")
    
    Y_row = np.arange(0,px)
    U_row = np.arange(px, px * 5 // 4)
    V_row = np.arange(px * 5 // 4, px * 3 // 2)
    Y_col = np.arange(0, f_num * px * 3 //2, px * 3 //2)
    U_col = np.arange(0, f_num * px * 3 //2, px * 3 //2)
    V_col = np.arange(0, f_num * px * 3 //2, px * 3 //2)
    print(Y_row.shape, U_row.shape, V_row.shape, Y_col.shape)
    Y_indices = matrix_row_col_add(Y_row,Y_col)
    Y_indices = np.reshape(Y_indices, (-1))
    print(Y_indices.shape)
    U_indices = matrix_row_col_add(U_row,U_col)
    U_indices = np.reshape(U_indices, (-1))
    V_indices = matrix_row_col_add(V_row,V_col)
    V_indices = np.reshape(V_indices, (-1))
    
    print(np.shape(Y_indices))
    print(np.shape(U_indices))
    print(np.shape(V_indices))
    
    Y = YUVarray[Y_indices]
    Y = Y.reshape((f_num,h,w))
    U = YUVarray[U_indices]
    U = U.reshape((f_num,h//2,w//2))
    V = YUVarray[V_indices]
    V = V.reshape((f_num,h//2,w//2))
    
    return Y, U, V

def YUV2RGB888(Y,U,V, yuvtype="420p", std="BT709"):
    """
    Y: 16 - 25; Cb/Cr: 16-240,
    
    Y: (h, w). 
    U: (h//2, w//2)
    V: (h//2, w//2)
    yuvtype: 420p, (to be added), 422p, 444p
    std: BT709, (to be added), BT601, ...
    
    return: 
    (h, w, 3) uint8 RGB
    """
    
    if yuvtype == "420p":
        U_full = matrix_1x122x2(U)
        V_full = matrix_1x122x2(V)
        
        R, G, B = YCbCr2RGB_ITU_BT709(Y.astype(np.float64), U_full.astype(np.float64), V_full.astype(np.float64))
        R = (R > 0.) * R
        G = (G > 0.) * G
        B = (B > 0.) * B
        R = (R <= 255.) * R + (R > 255.) * 255.
        G = (G <= 255.) * G + (G > 255.) * 255.
        B = (B <= 255.) * B + (B > 255.) * 255.
        
        R = np.array(R, dtype="uint8")
        G = np.array(G, dtype="uint8")
        B = np.array(B, dtype="uint8")
        RGB = np.concatenate([R[:,:,np.newaxis], G[:,:,np.newaxis], B[:,:,np.newaxis]], axis=2)

        return RGB

if __name__ == "__main__":
    row_label = ["1/50s", "1/30s"]
    col_label = ["30fps", "24fps"]
    Texts = [["",""],["",""]]
    # makeTable(row_label, col_label, Texts, "nHDR(8bit)")
 
    
    filepath = "/Volumes/Dirk_SSD/Sinan_10bit_1hz_Check_2022_4_27/"
    # for file in os.listdir("/Volumes/Dirk_SSD/Sinan_10bit_1hz_Check_2022_4_27/"):
    #     if not file.endswith(".csv") and not file.endswith(".yuv") and not file.startswith("._") and (file.endswith(".mp4") or file.endswith(".MOV")):      
    file = "Samsung_rear_nHDR_30fps_30s_1.mp4"
    print(file)
    forehead_ROI = selectROI(filepath + file, "forehead")
    leftCheak_ROI = selectROI(filepath + file, "leftCheak")
    rightCheak_ROI = selectROI(filepath + file, "rightCheak")
    Curtain1_ROI = selectROI(filepath + file, "Curtain1")
    Curtain2_ROI = selectROI(filepath + file, "Curtain2")
    # sio.savemat("ROIS/" + file[:-4] + ".mat", 
    #             {"forehead":forehead_ROI, "left" : leftCheak_ROI, "right": rightCheak_ROI, 
    #                 "curtain1": Curtain1_ROI, "curtain2": Curtain2_ROI})
            
            
            
    # YUV = np.fromfile("/Volumes/Dirk_SSD/Sinan_10bit_1hz_Check_2022_4_27/Samsung_rear_nHDR_30fps_30s_0_0.yuv", dtype="uint8")
    # vid = cv2.VideoCapture("/Volumes/Dirk_SSD/Sinan_10bit_1hz_Check_2022_4_27/Samsung_rear_nHDR_30fps_30s_0.mp4")
    # f_num = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(f_num)
    # success, img = vid.read()
    # h = img.shape[0]
    # w = img.shape[1]
    # Y, U, V = YUV420p2YUV(YUV, w, h, f_num)
    # np.save("Y.npy",Y)
    # np.save("U.npy",U)
    # np.save("V.npy",V)
    
def video_duration(filename):
    start = time.time()
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num / rate
        end = time.time()
        spend = end - start
        # print("获取视频时长方法3耗时：", spend)
        return frame_num,duration
    return -1

def peak_delay(a,b,gt_HR):
    peak_delay_time = 0
    peak_step = int(3600/gt_HR)
    a1 = a[0:peak_step]
    b1 = b[0:peak_step]
    peak_delay_time = np.argmax(a1)-np.argmax(b1)

    return peak_delay_time

def bottom_delay(a,b,gt_HR):
    bottom_delay_time = 0
    peak_step = int(3600/gt_HR)
    a1 = a[0:peak_step]
    b1 = b[0:peak_step]
    bottom_delay_time = np.argmin(a1)-np.argmin(b1)

    return bottom_delay_time
    


def cross_corr(s1,s2):
    c21 = scipy.signal.correlate(s2,s1,mode='full',method = 'auto')
    # c21 = np.correlate(s2,s1,mode='full')
    # print(c21)
    t21 = np.argmax(c21)
    len_s = len(s1)
    index = t21-len_s
    delay = index
# 若index>0，则说明s1信号领先s2信号index个距离
# 若index<0，则说明s2信号领先s1信号index个距离
    if index > 0 :
        tt1 = s2[index:]
        tt2 = s2[0:index]
        s2_0 = np.concatenate((tt1, tt2), axis=0)
    else:
        index = len_s + index
        tt1 = s2[0:index]
        tt2 = s2[index:]

        s2_0 = np.concatenate((tt2, tt1), axis=0)
    
    return s1,s2_0,delay
from turtle import st
import moviepy.editor as mp
import skvideo.io

import os
import subprocess 
import sys
from matplotlib import pyplot as plt
from moviepy.config import get_setting
from moviepy.tools import subprocess_call
import numpy as np
from scipy.fft import *
from scipy.io import wavfile
import wave
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip



def extract_audio(video,output):
    # if os.path.exists(output) is False:
    if 1==1:
        command = "ffmpeg -y -i "+str(video) +" -ar 16000 -ac 1 -f wav "+str(output)
        # command = "ffmpeg -n -i "+str(video) +" -ss 0.000 -t 10.000 -ar 16000 -ac 1 -f wav "+str(output)
        print(command)
        subprocess.call(command,shell=True)
    else:
        return 'already exist'

def extract_video(video,output,start_time):
    if os.path.exists(output) is False:
        # command = "ffmpeg -y -vsync 0 -hwaccel cuda -c:v h264_cuvid -i "+str(video) +" -ss "+str(start_time)+" -t 60 -an -c:a copy -vcodec h264_nvenc -keyint_min 2 -g 1 -y "+str(output)
        command = "ffmpeg -y -vsync 0 -hwaccel cuda -c:v h264_cuvid -i "+str(video) +" -ss "+str(start_time)+" -t 60 -an -c:a copy -keyint_min 2 -g 1 -y "+str(output)
        # command = "ffmpeg -n -i "+str(video) +" -ss 0.000 -t 10.000 -ar 16000 -ac 1 -f wav "+str(output)
        print(command)
        subprocess.call(command,shell=True)
    else:
        return 'already exist'

def draw_wav(filename):
    fig = np.memmap(filename, dtype='h', mode='r')
    # print(fig)
    plt.plot(fig)
    plt.show()
    return fig

def seconds_to_time(seconds):
    # seconds=5555.55
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    a = 1000*(s-int(s))
    time = ("%02d:%02d:%02d.%03d" % (h, m, s,a))
    # print (time)
    return time

def freq(file, start_time, end_time):

    # Open the file and convert to mono
    sr, data = wavfile.read(file)
    if data.ndim > 1:
        data = data[:, 0]
    else:
        pass

    # Return a slice of the data from start_time to end_time
    dataToRead = data[int(start_time * sr / 1000) : int(end_time * sr / 1000) + 1]

    # Fourier Transform
    N = len(dataToRead)
    yf = rfft(dataToRead)
    xf = rfftfreq(N, 1 / sr)

    # Uncomment these to see the frequency spectrum as a plot
    # plt.plot(xf, np.abs(yf))
    # plt.show()

    # Get the most dominant frequency and return it
    idx = np.argmax(np.abs(yf))
    freq = xf[idx]
    return freq

def get_wav_duration(wavefile):
    with wave.open(wavefile,'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            wav_length = frames / float(rate)
            #wav_length = round(frames / float(rate), 1)
            print("音频长度：",wav_length,"秒")

    wavetime = int(wav_length)*1000
    timestep = 25
    num_frame = 1000/timestep
    start_frame = 0
    end_frame = 0
    start_freq = 4000
    end_freq = 2000
    # tolerate = 0.01

    def frequency_in_range(freq_num, target):
        tolerate = 0.01
        if (freq_num<target*(1+tolerate) and freq_num >target*(1-tolerate)):
            return 1
        else:
            return 0

    for i in range(0,wavetime,timestep):
        frequency = freq(wavefile,i,i+timestep)
        frequency_before = frequency_after = 0
        if i/timestep > 0:
            frequency_before = freq(wavefile,i-timestep,i)
        if i< wavetime-timestep:
            frequency_after = freq(wavefile,i+timestep,i+2*timestep)
        
        # print(frequency)
        if(frequency_in_range(frequency,start_freq) == 1 and frequency_in_range(frequency_before,start_freq) == 1 ):
            start_frame = i/timestep
        if(frequency_in_range(frequency,end_freq) == 1 and frequency_in_range(frequency_after,end_freq) == 1):
            end_frame = i/timestep
            break
    print('start_frame :',start_frame)
    print('end_frame :',end_frame)
    t1 = (start_frame+1)/num_frame
    # t2 = t1 + 60
    t2 = (end_frame)/num_frame
    print('裁剪后的视频长度为：',t2-t1)
    return t1,t2






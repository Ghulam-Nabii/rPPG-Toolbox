import serial
import winsound
import sounddevice as sd
import threading
from scipy.signal import chirp,spectrogram
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave 
import time
import os

timestamp_record = [] 
def recording_sound(file_path):
    sec = 4 # s
    #创建对象
    pa = pyaudio.PyAudio()
    # with wave.open(file_path,'w') as wf:
    #创建流：采样位，声道数，采样频率，缓冲区大小，input
    stream = pa.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index= 1,
                    frames_per_buffer=1024)
    #创建式打开音频文件
    wf = wave.open(file_path, "wb")
    #设置音频文件的属性：采样位，声道数，采样频率，缓冲区大小，必须与流的属性一致
    wf.setnchannels(1)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    print("开始录音")
    #采样频率*秒数/缓冲区大小
    print('start record:',time.time())
    start_win_record_time = time.time()
    for w in range(int(16000*sec/1024)):
        data = stream.read(1024)#每次从流的缓存中读出数据
        wf.writeframes(data)#把读出的数据写入wf
    print('end record:',time.time())
    end_win_record_time = time.time()

    print("录音结束")
    
    stream.stop_stream()#先把流停止
    stream.close()#再把流关闭
    pa.terminate()#把对象关闭
    wf.close()#把声音文件关闭，否则不能播放
    timestamp_record.append(start_win_record_time)
    timestamp_record.append(end_win_record_time)
    return file_path

def start_sound_wav_out():
    fs = 16000
    t = np.linspace(0,3,fs)
    w = chirp(t, f0=2000, f1=200, t1=3, method='linear')
    start_sound_play_time =time.time()
    sd.play(w,fs)
    print('start play :',start_sound_play_time)
    timestamp_record.append(start_sound_play_time)
    return start_sound_play_time

def end_sound_wav_out():
    fs = 16000
    t = np.linspace(0,3,fs)
    w = chirp(t, f0=300, f1=3000, t1=3, method='linear')
    start_sound_play_time =time.time()
    sd.play(w,fs)
    print('end play :',start_sound_play_time)
    timestamp_record.append(start_sound_play_time)
    return start_sound_play_time


if __name__ == "__main__": 
    origin_path =  r'C:\Users\T JACK\Desktop\ACSP\code\GreenChannel\audio_test' #数据集根目录
    current_path = origin_path
    os.chdir(current_path)
    all_time = []
    for i in range(10) :
        timestamp_record = []
        start_file = str(i)+'start.wav'
        end_file = str(i)+'end.wav'
        print('--------------',i+1,'--------------')
        task1 = threading.Thread(target=recording_sound,args=(start_file,))
        task2 = threading.Thread(target=start_sound_wav_out)
        task3 = threading.Thread(target=recording_sound,args=(end_file,))
        task4 = threading.Thread(target=end_sound_wav_out)

        os.system('pause')
        time.sleep(1)
        task1.start()
        time.sleep(0.5)
        task2.start()
        time.sleep(6)

        task3.start()
        time.sleep(0.5)
        task4.start()
        time.sleep(5)
        print(timestamp_record)
        all_time.append(timestamp_record[4]-timestamp_record[1])
    print(all_time)
    all_time = np.array(all_time)
    np.save('all_time.npy',all_time)
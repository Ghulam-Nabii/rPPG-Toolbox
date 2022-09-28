import os
from select import select
from mp4_wav import mp4_cut
from utils import selectROI
import scipy.io as sio
import pathlib
import glob

#自动命名拍摄视频 
subject_id = 1

excel_path = str(subject_id)+'.xlsx'
origin_path =   r'E:\rPPG\dataset' #数据集根目录
current_path = os.path.join(origin_path,str(subject_id))#实验者目录
os.chdir(current_path)

def file_filter(f):
    if f[-4:] in ['.mp4']:
        return True
    else:
        return False


#
filelist = os.listdir(current_path)   # 文件夹路径
filelist = list(filter(file_filter, filelist))
filetype = '.mp4'             # 文件类型
filelist.sort(key=lambda x: int(x[-10: -4])) 
file_order = 0
print("filelist:",filelist)
for file in filelist:
    
    Olddir = os.path.join(current_path, file)
    if os.path.isdir(Olddir):
        continue
    filename = os.path.splitext(file)[0]
    print("filename1:",filename)
    filename = str(subject_id) # 重命名的规则
    print("filename2:",filename)
    filetype = os.path.splitext(file)[1]
    filename = str(filename)
    Newdir = os.path.join(current_path,  str(subject_id)+'_'+str(file_order)+ filetype) # zfill(6) # 填充到6位字符串
    print("filename3:",Newdir)
    os.rename(Olddir, Newdir)
    file_order+=1


filepath = str(subject_id)+"_0.mp4" #---------------TBD
# mp4_cut(filepath)
# filepath = "PPG_Front.mp4"
roi_names = ["Forehead", "left", "right", "curtain"] #---------------TBD can add or remove roi area
# roi_names = ["wide_forehead", "narrow_forehead", "nose_bridge", "nose",
#              "narrow_leftcheak", "wide_leftcheak", "narrow_rightcheak",
#              "wide_rightcheak", "philtrum", "wide_chin", "narrow_chin", "curtain1", "curtain2", "curtain3"] #

Roi_dict = {}


print("working on:", filepath)
mat_name = "ROI.mat" #---------------TBD

# if os.path.exists(mat_name):
#     continue




for i in range(len(roi_names)):
    Roi_dict[roi_names[i]] = selectROI(filepath, roi_names[i])

sio.savemat(mat_name, Roi_dict)
        
# for file in os.listdir("/Volumes/Dirk_SSD/Sinan_10bit_1hz_Check_2022_4_27/"):
#     if not file.endswith(".csv") and not file.endswith(".yuv") and not file.startswith("._") and (file.endswith(".mp4") or file.endswith(".MOV")):      
#         if os.path.exists("ROIS/" + file[:-4] + ".mat"):
#             continue
#         print(file)
#         for i in range(len(roi_names)):
#             Roi_dict[roi_names[i]] = selectROI(filepath + file, roi_names[i])
            
#         sio.savemat("ROIS/" + file[:-4] + ".mat", Roi_dict)
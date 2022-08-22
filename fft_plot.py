from neural_methods.trainer.BaseTrainer import BaseTrainer
import torch
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import logging
from metrics.metrics import calculate_metrics
from collections import OrderedDict
import glob
from math import ceil
import argparse
import glob
import os
import torch
import re
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset import data_loader
from eval.post_process import *
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.model.DeepPhys import DeepPhys
from collections import OrderedDict
import random
import numpy as np
import matplotlib.pyplot as plt



activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

def getitem(inputs,labels,index):
    """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
    data = np.load(inputs[index])
    label = np.load(labels[index])
    # data = np.transpose(data, (0, 3, 1, 2))
    data = np.transpose(data, (3, 0, 1, 2))  # physnet
    data = np.float32(data)
    label = np.float32(label)
    item_path = inputs[index]
    item_path_filename = item_path.split('/')[-1]
    split_idx = item_path_filename.index('_')
    filename = item_path_filename[:split_idx]
    chunk_id = item_path_filename[split_idx+6:].split('.')[0]
    return data, label, filename, chunk_id

def read_label(dataset):
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key,
                                                 value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr

def reform_data_from_dict(data):
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))


def cat_pred(predictions, labels):
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    label_hr = list()
    label_dict = read_label("UBFC")
    white_list = []
    for index in predictions.keys():
        if index in white_list:
            continue
        prediction = reform_data_from_dict(predictions[index])
        label = reform_data_from_dict(labels[index])
    return prediction, label

def calculate_metrics(predictions, labels):
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    label_hr = list()
    label_dict = read_label("UBFC")
    white_list = []
    for index in predictions.keys():
        if index in white_list:
            continue
        prediction = reform_data_from_dict(predictions[index])
        label = reform_data_from_dict(labels[index])
        gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
            prediction, label, fs=30)
        # print(predictions[i]['prediction'], labels[i]['prediction'])
        gt_hr_peak, pred_hr_peak = calculate_metric_peak_per_video(
            prediction, label, fs=30)
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)
        video_index, GT_HR = read_hr_label(label_dict, index)
        label_hr.append(GT_HR)
        if abs(GT_HR - pred_hr_fft) > 10:
            print('Video Index: ', video_index)
            print('GT HR: ', GT_HR)
            print('Pred HR: ', pred_hr_fft)
    predict_hr_peak_all = np.array(predict_hr_peak_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    label_hr_all_manual = np.array(label_hr)
    for metric in ['MAE', 'RMSE', 'MAPE', 'Pearson']:
        if metric == "MAE":
            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - label_hr_all_manual))
            MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - label_hr_all_manual))
            print("FFT MAE:{0}".format(MAE_FFT))
            print("Peak MAE:{0}".format(MAE_PEAK))

            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_peak_all))
            MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
            print("FFT MAE (Peak Label):{0}".format(MAE_FFT))
            print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))

            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
            MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_fft_all))
            print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            print("Peak MAE (FFT Label):{0}".format(MAE_PEAK))

        elif metric == "RMSE":
            RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - label_hr_all_manual)))
            RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - label_hr_all_manual)))
            print("FFT RMSE:{0}".format(RMSE_FFT))
            print("PEAK RMSE:{0}".format(RMSE_PEAK))

            RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_peak_all)))
            RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
            print("FFT RMSE (Peak Label):{0}".format(RMSE_FFT))
            print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))

            RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
            RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_fft_all)))
            print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            print("PEAK RMSE (FFT Label):{0}".format(RMSE_PEAK))

        elif metric == "MAPE":
            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - label_hr_all_manual) / label_hr_all_manual)) * 100
            MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - label_hr_all_manual) / label_hr_all_manual)) * 100
            print("FFT MAPE:{0}".format(MAPE_FFT))
            print("PEAK MAPE:{0}".format(MAPE_PEAK))

            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
            MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
            print("FFT MAPE (Peak Label):{0}".format(MAPE_FFT))
            print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))

            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
            MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
            print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            print("PEAK MAPE (FFT Label):{0}".format(MAPE_PEAK))

        elif metric == "Pearson":
            Pearson_FFT = np.corrcoef(predict_hr_fft_all, label_hr_all_manual)
            Pearson_PEAK = np.corrcoef(predict_hr_peak_all, label_hr_all_manual)
            print("FFT Pearson:{0}".format(abs(Pearson_FFT[1, 0])))
            print("PEAK Pearson:{0}".format(abs(Pearson_PEAK[1, 0])))
            # print("FFT Pearson:{0}".format(Pearson_FFT[0][1]))
            # print("PEAK Pearson:{0}".format(Pearson_PEAK[0][1]))

            Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_peak_all)
            Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
            print("FFT Pearson (Peak Label):{0}".format(Pearson_FFT[0][1]))
            print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))

            Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
            Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_fft_all)
            print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            print("PEAK Pearson (FFT Label):{0}".format(Pearson_PEAK[0][1]))

        else:
            raise ValueError("Wrong Test Metric Type")







if __name__ == "__main__":
    # cached_path = "/data2/rppg_datasets/PreprocessedData/UBF" \
    # "C_SizeW128_SizeH128_ClipLength128_DataTypeStandardized_L" \
    # "abelTypeStandardized_Large_boxTrue_Large_size1.5_D"\
    # "yamic_DetFalse_det_len180"
    # cached_path = "/data2/rppg_datasets/Preprocessed" \
    #               "Data/UBFC_SizeW128_SizeH128_ClipLength12" \
    #               "8_DataTypeRaw_LabelTypeRaw_Large_boxTrue_Larg" \
    #               "e_size1.5_Dyamic_DetFalse_det_len180/"
    cached_path = "/data2/rppg_datasets/PreprocessedData" \
                  "/UBFC_SizeW128_SizeH128_ClipLength128_Data" \
                  "TypeNormalized_LabelTypeNormalized_Large_boxT" \
                  "rue_Large_size1.5_Dyamic_DetFalse_det_len180/"
    # cached_path = "/data2/rppg_datasets/PreprocessedData/PURE_SizeW128_SizeH128_ClipLength128_DataTypeStandardized_LabelTypeStandardized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180_0.8_1.0/"
    inputs_data = glob.glob(os.path.join(cached_path, "subject33_input*.npy"))
    labels_data = [input.replace("input", "label") for input in inputs_data]
    assert (len(inputs_data) == len(labels_data))
    length = len(inputs_data)
    print("load lens:", length)

    cached_path2 = "/data2/rppg_datasets/PreprocessedData/UBF" \
    "C_SizeW128_SizeH128_ClipLength128_DataTypeStandardized_L" \
    "abelTypeStandardized_Large_boxTrue_Large_size1.5_D"\
    "yamic_DetFalse_det_len180"

    # cached_path2 = "/data2/rppg_datasets/Preprocessed" \
    #               "Data/UBFC_SizeW128_SizeH128_ClipLength12" \
    #               "8_DataTypeRaw_LabelTypeRaw_Large_boxTrue_Larg" \
    #               "e_size1.5_Dyamic_DetFalse_det_len180/"
    # cached_path2 = "/data2/rppg_datasets/PreprocessedData/PURE_SizeW128_SizeH128_ClipLength128_DataTypeNormalized_LabelTypeNormalized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180_0.8_1.0/"
    inputs_data2 = glob.glob(os.path.join(cached_path2, "subject33_input*.npy"))
    labels_data2 = [input.replace("input", "label") for input in inputs_data2]
    assert (len(inputs_data2) == len(labels_data2))
    length2 = len(inputs_data2)
    print("load lens:", length2)

    # Deepphys
    # model = DeepPhys(img_size=72).to("cuda")
    # model = torch.nn.DataParallel(model, device_ids=list(range(4)))
    # person_model_paths = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreTrainedModels/deepphys_synthetics_10epoch_geforce2080ti_Epoch9.pth"
    # model.load_state_dict(torch.load(person_model_paths, map_location=torch.device('cuda')))

    # TSCAN
    model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=128).to("cuda")  # [3, T, 128,128]
    # model = torch.nn.DataParallel(model, device_ids=list(range(1)))
    # person_model_paths = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreTrai" \
    #                      "nedModels/PURE_SizeW128_SizeH128_ClipLength128_DataTypeStandardi" \
    #                      "zed_LabelTypeStandardized_Large_boxTrue_Large_size1.5_Dyamic_DetFa" \
    #                      "lse_det_len180/PURE_PURE_UBFC_physnet.pth_Epoch11.pth"
    # person_model_paths ="/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreTrainedMod" \
    #                     "els/PURE_SizeW128_SizeH128_ClipLength128_DataTypeNormalized_LabelTypeN" \
    #                     "ormalized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180/PURE_PUR" \
    #                     "E_UBFC_physnet.pth_Epoch11.pth"
    # person_model_paths = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPP" \
    # "G-Toolbox/PreTrainedModels/PURE_SizeW128_SizeH12" \
    # "8_ClipLength128_DataTypeRaw_LabelTypeRaw_Large_boxT" \
    # "rue_Large_size1.5_Dyamic_DetFalse_det_len180/PURE_PURE_UBFC_physnet.pth_Epoch9.pth"
    person_model_paths = "/data1/acsp/Yuzhe_Zhang/Toolb" \
                         "ox_master_2/rPPG-Toolbox/PreTrainedModels/" \
                         "PURE_SizeW128_SizeH128_ClipLength128_DataTypeStand" \
                         "ardized_LabelTypeStandardized_Large_boxTrue_Large_size1." \
                         "5_Dyamic_DetFalse_det_len180/PURE_PURE_UBFC_physnet.pth_Epoch3.pth"
    model.load_state_dict(torch.load(person_model_paths, map_location=torch.device('cuda')))


    # model2 = DeepPhys(img_size=72).to("cuda")
    # model2 = torch.nn.DataParallel(model2, device_ids=list(range(4)))
    # person_model_paths2 = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreTrainedModels/PURE_SizeW72_SizeH72_ClipLength180_DataTypeNormalized_Standardized_LabelTypeNormalized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180/PURE_UBFC_deepphys.pth_Epoch3.pth"
    # model2.load_state_dict(torch.load(person_model_paths2, map_location=torch.device('cuda')))
    model.eval()
    # model2.eval()

    # model.attn_mask_1.register_forward_hook(get_activation('attn_mask_1'))
    # mask = activation['attn_mask_1'][0].view(34, 34)
    # mask = mask / torch.max(mask)

    predictions = dict()
    labels = dict()
    predictions2 = dict()
    labels2 = dict()
    with torch.no_grad():
        for idx in range(length):
            data, label, filename, chunk_id = getitem(inputs_data,labels_data,idx)
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            data_test, labels_test = data.to("cuda").unsqueeze(0), label.to("cuda").unsqueeze(0)
            pred_ppg_test, x_visual, x_visual3232, x_visual1616 = model(data_test)
            subj_index = filename
            sort_index = int(chunk_id)
            if subj_index not in predictions.keys():
                predictions[subj_index] = dict()
                labels[subj_index] = dict()
            predictions[subj_index][sort_index] = pred_ppg_test
            labels[subj_index][sort_index] = labels_test
            # plt.figure(figsize=(8,8), dpi=80)
            # plt.plot(pred_ppg_test.cpu().numpy())
            # plt.title("TSCAN:"+subj_index+str(sort_index))
            # plt.show()
        print(data_test)
        plt.figure(figsize=(8, 8), dpi=80)
        plt.imshow(np.transpose(data_test[0, :, 0, :, :].cpu().numpy(), (1, 2, 0)))
        plt.title("data1")
        plt.show()
        for idx in range(length2):
            data2, label2, filename2, chunk_id2 = getitem(inputs_data2,labels_data2,idx)
            data2 = torch.from_numpy(data2)
            label2 = torch.from_numpy(label2)
            data_test2, labels_test2 = data2.to("cuda").unsqueeze(0), label2.to("cuda").unsqueeze(0)
            pred_ppg_test2, _, _, _ = model(data_test2)
            subj_index2 = filename2
            sort_index2 = int(chunk_id2)
            if subj_index2 not in predictions2.keys():
                predictions2[subj_index2] = dict()
                labels2[subj_index2] = dict()
            predictions2[subj_index2][sort_index2] = pred_ppg_test2
            labels2[subj_index2][sort_index2] = labels_test2
        print(data_test2)
        plt.figure(figsize=(8, 8), dpi=80)
        plt.imshow(np.transpose(data_test2[0, :, 0, :, :].cpu().numpy(), (1, 2, 0)))
        plt.title("data2")
        plt.show()
    physnet_pred, label1 = cat_pred(predictions, labels)
    physnet_pred2, label2 = cat_pred(predictions2, labels2)
    print(physnet_pred.shape)


    plt.figure(figsize=(8, 8), dpi=80)
    plt.plot(physnet_pred.cpu().numpy())
    plt.title("Physnet1:")
    plt.show()
    plt.figure(figsize=(8, 8), dpi=80)
    plt.plot(label1.cpu().numpy())
    plt.title("Physnet1_label:")
    plt.show()
    plt.figure(figsize=(8, 8), dpi=80)
    plt.plot(physnet_pred2.cpu().numpy())
    plt.title("Physnet2:")
    plt.show()
    plt.figure(figsize=(8, 8), dpi=80)
    plt.plot(label2.cpu().numpy())
    plt.title("Physnet2_label:")
    plt.show()
    label_sub = label1 - label2

    plt.figure(figsize=(8, 8), dpi=80)
    plt.plot(label_sub.cpu().numpy())
    plt.title("label Sub:" )
    plt.show()

    gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
        physnet_pred, label1, fs=30)
    gt_hr_peak, pred_hr_peak = calculate_metric_peak_per_video(
        physnet_pred, label1, fs=30)

    gt_hr_fft2, pred_hr_fft2 = calculate_metric_per_video(
        physnet_pred2, label2, fs=30)
    gt_hr_peak2, pred_hr_peak2 = calculate_metric_peak_per_video(
        physnet_pred2, label2, fs=30)
    print(subj_index)
    print("physnet_fft:",pred_hr_fft)
    print("gt_fft:",gt_hr_fft)
    print("tscan_peak:",pred_hr_peak)
    print("gt_peak:",gt_hr_peak)

    print(subj_index2)
    print("physnet_fft2:",pred_hr_fft2)
    print("gt_fft2:",gt_hr_fft2)
    print("physnet_peak2:",pred_hr_peak2)
    print("gt_peak2:",gt_hr_peak2)
    # gt_hr_fft, pred_hr_fft_2 = calculate_metric_per_video(
    #     deep_pred, label2, fs=30)
    # gt_hr_peak, pred_hr_peak = calculate_metric_peak_per_video(
    #     deep_pred, label2, fs=30)
    # print("deepphys_fft:",pred_hr_fft_2)
    # print("deepphys_peak:",pred_hr_peak)

    calculate_metrics(predictions, labels)



    # cached_path = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreprocessedData/UBFC_SizeW72_SizeH72_ClipLength180_DataTypeNormalized_Standardized_LabelTypeNormalized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180/"
    print("success!")

    cached_path = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/Preprocess" \
                  "edData/UBFC_SizeW72_SizeH72_ClipLength180_DataTypeNor" \
                  "malized_Standardized_LabelTypeNormalized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180/"
    inputs_data_con = glob.glob(os.path.join(cached_path, "subject8_input*.npy"))
    labels_data_con = [input.replace("input", "label") for input in inputs_data_con]
    assert (len(inputs_data_con) == len(labels_data_con))
    length_2 = len(inputs_data_con)
    print("load lens:", length_2)

    model2 = DeepPhys(img_size=72).to("cuda")
    model2 = torch.nn.DataParallel(model2, device_ids=list(range(4)))
    person_model_paths2 = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreTrainedModels/PURE_SizeW72_SizeH72_ClipLength180_DataTypeNormalized_Standardized_LabelTypeNormalized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180/PURE_UBFC_deepphys.pth_Epoch3.pth"
    model2.load_state_dict(torch.load(person_model_paths2, map_location=torch.device('cuda')))

    predictions_con = dict()
    labels_con = dict()
    with torch.no_grad():
        for idx in range(length_2):
            data, label, filename, chunk_id = getitem(inputs_data_con,labels_data_con,idx)
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            data_test, labels_test = data.to("cuda"), label.to("cuda")
            D, C, H, W = data_test.shape
            data_test = data_test.contiguous().view(D, C, H, W)
            labels_test = labels_test.contiguous().view(-1, 1)
            pred_ppg_test = model2(data_test)
            subj_index = filename
            sort_index = int(chunk_id)
            if subj_index not in predictions_con.keys():
                predictions_con[subj_index] = dict()
                labels_con[subj_index] = dict()
            predictions_con[subj_index][sort_index] = pred_ppg_test
            labels_con[subj_index][sort_index] = labels_test
    calculate_metrics(predictions_con, labels_con)



# for name, layer in model.named_modules():
# ...     if isinstance(layer, torch.nn.Conv2d):
# ...             print(name, layer)
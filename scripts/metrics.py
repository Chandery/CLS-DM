from pathlib import Path
import SimpleITK as sitk
import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch

import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ldm.util import AverageMeter


import nibabel as nib
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as SSIM
from scipy.spatial.distance import cosine
import numpy as np


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    L2_distance = torch.cdist(total, total) ** 2
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def load_nii(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def calculate_mmd(data1, data2):
    return np.mean(np.square(data1 - data2))

def normalization(data):
    return (data-data.min())/(data.max()-data.min())

def Peak_Signal_to_Noise_Rate(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    :param arr1:
      Format-[NDHW], OriImage [0,1]
    :param arr2:
      Format-[NDHW], ComparedImage [0,1]
    :return:
      Format-None if size_average else [N]
    """
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    if len(arr1.shape) == 3:
        arr1 = np.expand_dims(arr1, axis=0)
    if len(arr2.shape) == 3:
        arr2 = np.expand_dims(arr2, axis=0)

    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    # Depth
    mse_d = (
        se.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(2)
    )
    zero_mse = np.where(mse_d == 0)
    mse_d[zero_mse] = eps
    psnr_d = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_d))
    # #zero mse, return 100
    psnr_d[zero_mse] = 100
    psnr_d = psnr_d.mean(1)

    # Height
    mse_h = (
        se.mean(axis=1, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(1)
    )
    zero_mse = np.where(mse_h == 0)
    mse_h[zero_mse] = eps
    psnr_h = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_h))
    # #zero mse, return 100
    psnr_h[zero_mse] = 100
    psnr_h = psnr_h.mean(1)

    # Width
    mse_w = (
        se.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True).squeeze(2).squeeze(1)
    )
    zero_mse = np.where(mse_w == 0)
    mse_w[zero_mse] = eps
    psnr_w = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_w))
    # #zero mse, return 100
    psnr_w[zero_mse] = 100
    psnr_w = psnr_w.mean(1)

    psnr_avg = (psnr_h + psnr_d + psnr_w) / 3
    if size_average:
        return psnr_d.mean(), psnr_h.mean(), psnr_w.mean(), psnr_avg.mean()
    else:
        return psnr_d, psnr_h, psnr_w, psnr_avg

def Structural_Similarity(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    :param arr1:
      Format-[NDHW], OriImage [0,1]
    :param arr2:
      Format-[NDHW], ComparedImage [0,1]
    :return:
      Format-None if size_average else [N]
    """
    assert (len(arr1.shape)<=5) and (len(arr2.shape)<=5)
    if len(arr1.shape) == 5:
        arr1 = arr1[0]
    if len(arr2.shape) == 5:
        arr2 = arr2[0]
    if not isinstance(arr1, np.ndarray):
        arr1 = arr1.cpu().to(torch.float32).numpy()
    if not isinstance(arr2, np.ndarray):
        arr2 = arr2.cpu().to(torch.float32).numpy()
    if len(arr1.shape) == 3:
        arr1 = np.expand_dims(arr1, axis=0)
    if len(arr2.shape) == 3:
        arr2 = np.expand_dims(arr2, axis=0)
    
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[1]
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = SSIM(arr1_d[0][i], arr2_d[0][i], data_range=PIXEL_MAX, multichannel=True)
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)

    # Height
    arr1_h = np.transpose(arr1, (0, 3, 1, 2))
    arr2_h = np.transpose(arr2, (0, 3, 1, 2))
    ssim_h = []
    for i in range(N):
        ssim = SSIM(arr1_h[0][i], arr2_h[0][i], data_range=PIXEL_MAX, multichannel=True)
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)

    # Width
    arr1_w = np.transpose(arr1, (0, 1, 2, 3))
    arr2_w = np.transpose(arr2, (0, 1, 2, 3))
    ssim_w = []
    for i in range(N):
        ssim = SSIM(arr1_w[0][i], arr2_w[0][i], data_range=PIXEL_MAX, multichannel=True)
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return ssim_d.mean(), ssim_h.mean(), ssim_w.mean(), ssim_avg.mean()
    else:
        return ssim_d, ssim_h, ssim_w, ssim_avg

def calculate_metrics(file1, file2):
    data1 = load_nii(file1)
    data2 = load_nii(file2)
    data1 = data1 / 255 * 2500
    data2 = data2 / 255 * 2500
    
    ssim_value_0, ssim_value_1, ssim_value_2, ssim_value_avg= Structural_Similarity(data1, data2, size_average=True, PIXEL_MAX=4095)
    ssim_value = [ssim_value_0, ssim_value_1, ssim_value_2, ssim_value_avg]
    ssim_value.append(SSIM(data1, data2, data_range=4095))

    mse_value = mean_squared_error(data1, data2)
    mae_value = np.mean(np.abs(data1 - data2))

    psnr_value_0, psnr_value_1, psnr_value_2, psnr_value_avg = Peak_Signal_to_Noise_Rate(data1, data2, size_average=True, PIXEL_MAX=4095)
    psnr_value = [psnr_value_0, psnr_value_1, psnr_value_2, psnr_value_avg]
    psnr_value.append(psnr(data1, data2, data_range=4095))

    cosine_similarity = 1 - cosine(data1.flatten(), data2.flatten())
    mmd_value = calculate_mmd(data1, data2)

    data1_norm = normalization(data1)
    data2_norm = normalization(data2)
    mse0_value = mean_squared_error(data1_norm, data2_norm)
    mae0_value = np.mean(np.abs(data1_norm - data2_norm))

    
    return ssim_value, mse_value, mae_value, psnr_value, cosine_similarity, mmd_value, mse0_value, mae0_value

def save_metrics_to_csv(metrics_dict, output_path):
    # 创建DataFrame，将字典转换为竖排形式
    df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存到CSV
    df.to_csv(output_path, index=False)
    print(f"指标已保存到: {output_path}")

if __name__ == "__main__":
    data_path = "path"
    # psnr_record_pl = AverageMeter()
    # ssim_record_pl = AverageMeter()
    psnr_d_pl = AverageMeter()
    psnr_h_pl = AverageMeter()
    psnr_w_pl = AverageMeter()
    psnr_avg_pl = AverageMeter()
    psnr_3d_pl = AverageMeter()

    ssim_d_pl = AverageMeter()
    ssim_h_pl = AverageMeter()
    ssim_w_pl = AverageMeter()
    ssim_avg_pl = AverageMeter()
    ssim_3d_pl = AverageMeter()
    mmd_record_pl = AverageMeter()

    psnr_pl = PeakSignalNoiseRatio()
    ssim_pl = StructuralSimilarityIndexMeasure()

    # ori_mhd_list = sorted(Path(data_path).glob("*origin*.nii"))
    # recon_mhd_list = sorted(Path(data_path).glob("*rec*.nii"))
    # ori_mhd_list = sorted(Path(data_path).glob("*origin*.mhd"))
    # recon_mhd_list = sorted(Path(data_path).glob("*reconstructions*.mhd"))

    ori_mhd_list = sorted(Path(data_path).glob("*origin*.nii.gz"))

    # recon_mhd_list = sorted(Path(data_path).glob("*ae_rec*.nii.gz"))
    # recon_mhd_list = sorted(Path(data_path).glob("*[!_ae]_rec.nii.gz"))
    recon_mhd_list = sorted(Path(data_path).glob("*rec.nii.gz"))
    
    recon_mhd_list_cond = sorted(f for f in recon_mhd_list if "cond" in f.stem)
    recon_mhd_list_ae = sorted(f for f in recon_mhd_list if "ae" in f.stem)
    recon_mhd_list_rec = sorted(f for f in recon_mhd_list if "rec" in f.stem and "ae" not in f.stem)

    recon_mhd_list = recon_mhd_list_rec  # ? ae or cond or rec
    
    mae_record_pl = AverageMeter()
    mse_record_pl = AverageMeter()
    mae0_record_pl = AverageMeter()
    mse0_record_pl = AverageMeter()
    cos_record_pl = AverageMeter()

    for ori, recon in tqdm.tqdm(zip(ori_mhd_list, recon_mhd_list), total=len(ori_mhd_list)):
        print(ori, '\n',recon)
        ssim_value, mse_value, mae_value, psnr_value, cosine_similarity, mmd_value, mse0_value, mae0_value= calculate_metrics(ori, recon)
        print(ssim_value)

        ssim_d_pl.update(ssim_value[0])
        ssim_h_pl.update(ssim_value[1])
        ssim_w_pl.update(ssim_value[2])
        ssim_avg_pl.update(ssim_value[3])
        ssim_3d_pl.update(ssim_value[4])

        psnr_d_pl.update(psnr_value[0])
        psnr_h_pl.update(psnr_value[1])
        psnr_w_pl.update(psnr_value[2])
        psnr_avg_pl.update(psnr_value[3])
        psnr_3d_pl.update(psnr_value[4])

        mmd_record_pl.update(mmd_value)
        mae_record_pl.update(mae_value)
        mse_record_pl.update(mse_value)
        cos_record_pl.update(cosine_similarity)
        mae0_record_pl.update(mae0_value)
        mse0_record_pl.update(mse0_value)
    
    print(f"PSNR-d mean±std:{psnr_d_pl.mean}±{psnr_d_pl.std}")
    print(f"PSNR-h mean±std:{psnr_h_pl.mean}±{psnr_h_pl.std}")
    print(f"PSNR-w mean±std:{psnr_w_pl.mean}±{psnr_w_pl.std}")
    print(f"PSNR-avg mean±std:{psnr_avg_pl.mean}±{psnr_avg_pl.std}")
    print(f"PSNR-3d mean±std:{psnr_3d_pl.mean}±{psnr_3d_pl.std}")

    print(f"SSIM-d mean±std:{ssim_d_pl.mean}±{ssim_d_pl.std}")
    print(f"SSIM-h mean±std:{ssim_h_pl.mean}±{ssim_h_pl.std}")
    print(f"SSIM-w mean±std:{ssim_w_pl.mean}±{ssim_w_pl.std}")
    print(f"SSIM-avg mean±std:{ssim_avg_pl.mean}±{ssim_avg_pl.std}")
    print(f"SSIM-3d mean±std:{ssim_3d_pl.mean}±{ssim_3d_pl.std}")

    print(f"MAE mean±std:{mae_record_pl.mean}±{mae_record_pl.std}")
    print(f"MSE mean±std:{mse_record_pl.mean}±{mse_record_pl.std}")
    print(f"MAE0 mean±std:{mae0_record_pl.mean}±{mae0_record_pl.std}")
    print(f"MSE0 mean±std:{mse0_record_pl.mean}±{mse0_record_pl.std}")
    print(f"CosineSimilarity mean±std:{cos_record_pl.mean}±{cos_record_pl.std}")

    # 创建指标字典
    metrics_dict = {
        'PSNR_d': f"{psnr_d_pl.mean:.4f}±{psnr_d_pl.std:.4f}",
        'PSNR_h': f"{psnr_h_pl.mean:.4f}±{psnr_h_pl.std:.4f}",
        'PSNR_w': f"{psnr_w_pl.mean:.4f}±{psnr_w_pl.std:.4f}",
        'PSNR_avg': f"{psnr_avg_pl.mean:.4f}±{psnr_avg_pl.std:.4f}",
        'PSNR_3d': f"{psnr_3d_pl.mean:.4f}±{psnr_3d_pl.std:.4f}",
        'SSIM_d': f"{ssim_d_pl.mean:.4f}±{ssim_d_pl.std:.4f}",
        'SSIM_h': f"{ssim_h_pl.mean:.4f}±{ssim_h_pl.std:.4f}",
        'SSIM_w': f"{ssim_w_pl.mean:.4f}±{ssim_w_pl.std:.4f}",
        'SSIM_avg': f"{ssim_avg_pl.mean:.4f}±{ssim_avg_pl.std:.4f}",
        'SSIM_3d': f"{ssim_3d_pl.mean:.4f}±{ssim_3d_pl.std:.4f}",
        'MAE': f"{mae_record_pl.mean:.4f}±{mae_record_pl.std:.4f}",
        'MSE': f"{mse_record_pl.mean:.4f}±{mse_record_pl.std:.4f}",
        'MAE0': f"{mae0_record_pl.mean:.4f}±{mae0_record_pl.std:.4f}",
        'MSE0': f"{mse0_record_pl.mean:.4f}±{mse0_record_pl.std:.4f}",
        'CosineSimilarity': f"{cos_record_pl.mean:.4f}±{cos_record_pl.std:.4f}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_path': data_path
    }

    # 保存到CSV
    output_path = os.path.join(data_path, 'metrics_results.csv')
    save_metrics_to_csv(metrics_dict, output_path)
         



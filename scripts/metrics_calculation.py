from pathlib import Path
import SimpleITK as sitk
import tqdm

# from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from monai.metrics.regression import SSIMMetric
from monai.metrics import PSNRMetric
import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ldm.util import AverageMeter


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


if __name__ == "__main__":
    # data_path = "/disk/cyq/2024/My_Proj/Xray-Diffusion/logs/autoencoder/pl_test_autoencoder-2024-09-18/13-27-09"
    data_path = "/disk/cc/Xray-Diffsuion/logs/autoencoder/pl_test_autoencoder-2024-10-29/09-45-14"
    psnr_record_pl = AverageMeter()
    ssim_record_pl = AverageMeter()
    mmd_record_pl = AverageMeter()

    # psnr_pl = PeakSignalNoiseRatio()
    # ssim_pl = StructuralSimilarityIndexMeasure()
    psnr_pl = PSNRMetric(max_val=255)
    ssim_pl = SSIMMetric(spatial_dims=3,data_range=255)

    # ori_mhd_list = sorted(Path(data_path).glob("*origin*.mhd"))
    # recon_mhd_list = sorted(Path(data_path).glob("*reconstructions*.mhd"))

    ori_mhd_list = sorted(Path(data_path).glob("*origin*.nii.gz"))
    
    print(ori_mhd_list)
    recon_mhd_list = sorted(Path(data_path).glob("*[!_ae]_rec.nii.gz"))
    print(recon_mhd_list)
    # recon_mhd_list = sorted(Path(data_path).glob("*ae_rec.nii.gz"))


    whole_recon = []
    whole_ori = []
    for ori, recon in tqdm.tqdm(zip(ori_mhd_list, recon_mhd_list), total=len(ori_mhd_list)):
        ori_img = sitk.ReadImage(str(ori))
        ori_arr = sitk.GetArrayFromImage(ori_img)
        ori_arr = torch.tensor(ori_arr)
        ori_arr = ori_arr.unsqueeze(0).unsqueeze(0)
        # ori_arr = ori_arr.view(1,128,128,128)

        recon_img = sitk.ReadImage(str(recon))
        recon_arr = sitk.GetArrayFromImage(recon_img)
        recon_arr = torch.tensor(recon_arr)
        recon_arr = recon_arr.unsqueeze(0).unsqueeze(0)
        # recon_arr = recon_arr.view(1,128,128,128)

        # whole_recon = recon_arr if len(whole_recon) == 0 else torch.cat((whole_recon, recon_arr), dim=0)
        # whole_ori = ori_arr if len(whole_ori) == 0 else torch.cat((whole_ori, ori_arr), dim=0)

        psnr = psnr_pl(recon_arr, ori_arr)
        ssim = ssim_pl(recon_arr, ori_arr)

        psnr_record_pl.update(psnr)
        ssim_record_pl.update(ssim)

    # whole_ori = whole_ori.permute(1, 0, 2, 3, 4).view(whole_ori.shape[0], -1)
    # whole_recon = whole_recon.permute(1, 0, 2, 3, 4).view(whole_recon.shape[0], -1)
    # mmd_score = mmd(whole_recon, whole_ori)

    print(f"PSNR mean±std:{psnr_record_pl.mean}±{psnr_record_pl.std}")
    print(f"SSIM mean±std:{ssim_record_pl.mean}±{ssim_record_pl.std}")
    # print(f"MMD:{mmd_score}")

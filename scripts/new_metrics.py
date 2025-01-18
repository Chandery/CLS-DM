from pathlib import Path
import SimpleITK as sitk
import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
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
    # data_path = "/disk/cyq/2024/My_Proj/VQGAN-DDPM/logs/cddpm/pl_test_cddpm-2024-06-24/15-07-57"
    # data_path = "/disk/cyq/2024/My_Proj/VQGAN-DDPM/logs/ldm/pl_test_ldm-2024-06-14/10-01-33"
    # data_path = "/disk/cyq/2024/My_Proj/VQGAN-DDPM/logs/c_vqgan_transformer/pl_test_transformer-2024-07-03/20-51-05"
    # data_path = "/disk/cc/Xray-Diffsuion/logs/autoencoder/pl_test_autoencoder-2024-10-30-fei/20-12-44"
    # new_origin = "/disk/cc/Xray-Diffsuion/logs/ldm/pl_test_ldm-2024-11-08/21-40-23/origin_moved_from_ae"
    data_path = "/disk/cc/Xray-Diffsuion/logs/ldm/pl_test_ldm-2024-11-13/18-14-24-one-previt"
    psnr_record_pl = AverageMeter()
    ssim_record_pl = AverageMeter()
    mmd_record_pl = AverageMeter()

    psnr_pl = PeakSignalNoiseRatio()
    ssim_pl = StructuralSimilarityIndexMeasure()

    # ori_mhd_list = sorted(Path(data_path).glob("*origin*.nii"))
    # recon_mhd_list = sorted(Path(data_path).glob("*reconstructions*.nii"))
    # ori_mhd_list = sorted(Path(data_path).glob("*origin*.mhd"))
    # recon_mhd_list = sorted(Path(data_path).glob("*reconstructions*.mhd"))

    ori_mhd_list = sorted(Path(data_path).glob("*origin*.nii.gz"))
    # recon_mhd_list = sorted(Path(data_path).glob("*ae_rec*.nii.gz"))
    # recon_mhd_list = sorted(Path(data_path).glob("*origin*.nii.gz"))
    recon_mhd_list = sorted(Path(data_path).glob("*[!_ae]_rec.nii.gz"))
    print(recon_mhd_list)
    whole_recon = []
    whole_ori = []
    for ori, recon in tqdm.tqdm(zip(ori_mhd_list, recon_mhd_list), total=len(ori_mhd_list)):
        ori_img = sitk.ReadImage(str(ori))
        ori_arr = sitk.GetArrayFromImage(ori_img)
        ori_arr = torch.tensor(ori_arr).to(torch.float32)
        ori_arr = ori_arr[None, None,::]

        recon_img = sitk.ReadImage(str(recon))
        recon_arr = sitk.GetArrayFromImage(recon_img)
        recon_arr = torch.tensor(recon_arr).to(torch.float32)
        recon_arr = recon_arr[None, None,::]

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

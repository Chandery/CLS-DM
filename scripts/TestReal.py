import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(z, c, proj_head, temperature=0.1):
    # 归一化处理
    z = F.normalize(z, p=2, dim=1)
    c = F.normalize(c, p=2, dim=1)
    
    c = proj_head(c)

    device = z.device

    # 信息保存正则项
    recon_loss = F.mse_loss(c, z)
    
    # 将 c 和 z 展平为二维张量
    c_flat = c.view(c.size(0), -1)
    z_flat = z.view(z.size(0), -1)

    # 对比项
    logits = torch.mm(z_flat, c_flat.T) / temperature
    labels = torch.arange(z.size(0)).to(device)
    return F.cross_entropy(logits, labels) + 0.5*recon_loss
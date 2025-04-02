import torch
import numpy as np
import os
import sys
import hydra
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ldm.CLIPAutoEncoder.autoencoder import CLIPAE
from dataset.monai_nii_dataset1 import AlignDataSet
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

def normalize(c):
    # tensor 归一化 & 排除nan   
    c = c.detach().cpu().numpy()
    if c.max() == c.min():
        c = np.zeros_like(c)
    else:
        c = (c - c.min()) / (c.max() - c.min())
    c = torch.from_numpy(c)
    return c

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    device = torch.device("cuda:0")
    ckpt_path = "/home/cdy/SharedSpaceLDM/ckpt/clipae-1248-ckpt/pl_train_autoencoder-epoch669-val_rec_loss227305.078.ckpt"
    pretrained_model = CLIPAE(save_path=config.hydra_path, config=config)
    pretrained_model.init_from_ckpt(ckpt_path)
    pretrained_model.eval()
    pretrained_model.to(device)
    print("pretrained_model loaded")

    original_model = CLIPAE(save_path=config.hydra_path, config=config)
    # kaiming init
    for m in original_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    original_model.eval()
    original_model.to(device)
    print("original_model loaded")

    test_ds = AlignDataSet(config, split = "dis")
    test_dl = DataLoader(
        dataset=test_ds,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        batch_size=1,
    )
    print("test_dl loaded")
    
    # 创建列表来存储所有输出
    all_output1 = []
    all_output2 = []
    all_z = []
    
    for batch_idx, batch in tqdm(enumerate(test_dl),total=test_ds.__len__()):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        reconstructions, _, z = pretrained_model(batch["image"])

        c = pretrained_model.get_cond(batch, type=config.cond_type)
        output1, _ = pretrained_model.conditional_encode(c)
        output2, _ = original_model.conditional_encode(c)
        output1 = pretrained_model.proj_head(output1)
        output2 = original_model.proj_head(output2)

        # z = normalize(z)
        # output1 = normalize(output1)
        # output2 = normalize(output2)
        # print(f"z:{z.min()}~{z.max()}({z.mean()},{z.std()}),output1:{output1.min()}~{output1.max()}({output1.mean()},{output1.std()}),output2:{output2.min()}~{output2.max()}({output2.mean()},{output2.std()})")
        
        # 收集每个batch的输出
        all_output1.append(output1.detach().cpu())
        all_output2.append(output2.detach().cpu())
        all_z.append(z.detach().cpu())
    
    # 合并所有batch的输出
    output1_all = torch.cat(all_output1, dim=0)
    output2_all = torch.cat(all_output2, dim=0)
    z_all = torch.cat(all_z, dim=0)
    # 转换为numpy数组并reshape
    output1_np = output1_all.numpy().reshape(output1_all.shape[0], -1)
    output2_np = output2_all.numpy().reshape(output2_all.shape[0], -1)
    z_np = z_all.numpy().reshape(z_all.shape[0], -1)
    # 合并两个模型的输出
    combined_outputs = np.concatenate((output1_np, output2_np, z_np), axis=0)
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=100)
    tsne_results = tsne.fit_transform(combined_outputs)

    z_res = tsne_results[output1_np.shape[0]+output2_np.shape[0]:, :]
    output1_res = tsne_results[:output1_np.shape[0], :]
    output2_res = tsne_results[output1_np.shape[0]:output1_np.shape[0]+output2_np.shape[0], :]

    np.save("plt/699_z_res.npy", z_res)
    np.save("plt/699_output1_res.npy", output1_res)
    np.save("plt/699_output2_res.npy", output2_res)
    
    # 绘制t-SNE结果
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:output1_np.shape[0], 0], tsne_results[:output1_np.shape[0], 1], 
               label='c space after Contrastive Learning', alpha=0.5)
    plt.scatter(tsne_results[output1_np.shape[0]:output1_np.shape[0]+output2_np.shape[0], 0], tsne_results[output1_np.shape[0]:output1_np.shape[0]+output2_np.shape[0], 1], 
               label='c space without Contrastive Learning', alpha=0.5)
    plt.scatter(tsne_results[output1_np.shape[0]+output2_np.shape[0]:, 0], tsne_results[output1_np.shape[0]+output2_np.shape[0]:, 1], 
               label='latent space', alpha=0.5)
    plt.legend()
    plt.title('t-SNE of All Outputs')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.savefig('tsne_distribution.png')
    plt.close()


if __name__ == "__main__":
    main()
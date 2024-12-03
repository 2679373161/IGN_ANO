from ING_PRO.mvtec2d import MVTec2D
from torchvision import transforms as T
from torch.utils.data import  ConcatDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ING_PRO.AEUnet_SAM import ReconstructiveSubNetwork as ReconstructiveSubNetworkOri
from ING_PRO.loss import SSIM
from torch import Tensor
import torch
from torch import optim
from ING_PRO.loss import SSIM
import os
import numpy as np
from typing import Any, List, Optional, Tuple, Union
import torch.nn.functional as fn
import kornia
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torchvision.transforms.functional as TF
import random
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    """
    init weights.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def data_loader_my():
    img_transform = T.Compose([T.Resize((256, 256)),
                T.CenterCrop(256),
                T.ToTensor(),
                ])

    mask_transform = T.Compose([T.Resize(256),
                T.CenterCrop(256),
                T.ToTensor(),
                ])
    train_data_transform = img_transform, mask_transform

    dataset = MVTec2D(data_path='D:/Myself/Postgraduate/07_Paper/Python_Prj/MLDFR/MVTec',
                        learning_mode='centralized',
                        phase='train',
                        data_transform=train_data_transform,
                        num_task=1)
    dataloader = DataLoader(dataset, batch_size=3, num_workers=16,
                            sampler=SubsetRandomSampler(dataset.sample_indices_in_task[0]), drop_last=False)
    return dataloader

model_Rec = ReconstructiveSubNetworkOri(in_channels=3, out_channels=3)
model_Rec_copy = ReconstructiveSubNetworkOri(in_channels=3, out_channels=3).requires_grad_(False)
model_Rec.to(device)
model_Rec_copy.to(device)
model_Rec.apply(weights_init)
model_Rec.load_state_dict(torch.load(os.path.join('./Ano_Try_1', f"epoch_200.pckl"), map_location='cuda:0'))

def model_train(dataloader):
    num_epochs = 200 #训练参数
    lastepoch = 0 #上次训练参数
    #参数初始化
    optimizer = torch.optim.Adam([{"params": model_Rec.parameters(), "lr": 0.0001}, ], weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [(num_epochs+lastepoch) * 0.8, (num_epochs+lastepoch) * 0.9], gamma=0.2, last_epoch=-1)
    rec_weight = 20
    idem_weight = 20
    tight_weight = 2.5
    idem_weight /= rec_weight
    tight_weight /= rec_weight
    loss_tight_clamp_ratio = 1.5
    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()

    n_iter = 0
    kernel = torch.ones(3, 3).to(device)

    for e in range(num_epochs):
        model_Rec.train()
        train_loss = 0.0
        epoch = e + lastepoch  + 1
        print("Epoch: " + str(epoch))
        for i_batch, sample_batched in enumerate(dataloader):
            # print("i_batch: " + str(i_batch))
            
            gray_batch = sample_batched["img"].clone().to(device)
            bsz = gray_batch.shape[0]
            z = torch.stack([Get_feature_map(gray_batch[I_Index],kernel) for I_Index in range(bsz)])
            z = z.to(device, memory_format=torch.contiguous_format)
            
            model_Rec_copy.load_state_dict(model_Rec.state_dict())
            fx = model_Rec(gray_batch)
            fz = model_Rec(z)
            f_z = fz.detach()
            ff_z = model_Rec(f_z)
            f_fz = model_Rec_copy(fz)
            # compute losses
            loss_rec = fn.huber_loss(fx, gray_batch, reduction="none").view(bsz, -1).mean(dim=-1)
            loss_idem = fn.huber_loss(f_fz, fz, reduction="mean")
            loss_tight = -fn.huber_loss(ff_z, f_z, reduction="none").view(bsz, -1).mean(dim=-1)
            loss_tight_clamp = loss_tight_clamp_ratio * loss_rec
            loss_tight = fn.tanh(loss_tight / loss_tight_clamp) * loss_tight_clamp
            loss_rec = loss_rec.mean()
            loss_tight = loss_tight.mean()
            
            ssim_loss = loss_ssim(fx, gray_batch)
            l2_loss = loss_l2(ff_z, gray_batch)
            
            
            loss = loss_rec + idem_weight * loss_idem + tight_weight * loss_tight+ ssim_loss +l2_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bsz
            n_iter += 1
        train_loss /= len(sample_batched["img"])
        print(f"Epoch {epoch} loss: {train_loss:.4f}")
        scheduler.step()
        if epoch % 20 == 0 or e == num_epochs - 1:
            torch.save(model_Rec.state_dict(), os.path.join('/root/autodl-fs/Ano_Try_1', f"epoch_{epoch}.pckl"))
    print("Train_ok")

def model_test(dataloader):
    kernel = torch.ones(3, 3).to(device)
    efficient_ti_model = build_efficient_sam_vitt()
    model_Rec.train()
    i = 0
    # for i_batch, sample_batched in enumerate(dataloader):
    sample_batched = dataloader.dataset[i]
    gray_batch = sample_batched["img"].clone()
    z  = efficient_ti_model.get_image_embeddings(
            gray_batch[None, ...]
        ).detach()
    z = z.to(device, memory_format=torch.contiguous_format)
    # # 根据梯度强度创建畸变权重
    # distortion_weight = torch.exp(-z[0]**2 / (2 * (0.1**2)))  # 可调整的高斯分布参数
    # # 生成较大的随机畸变
    # distortion = torch.randn_like(gray_batch[0]) * 0.9  # 可调整的畸变强度
    # # 在原始图像上应用畸变权重
    # corrupted_image_noise = distortion_weight * distortion
    # corrupted_image_noise = torch.clamp(corrupted_image_noise, 0, 1).unsqueeze(0)
    # # 确定损坏区域的位置和大小
    # top_left = (50, 50)  # 左上角坐标
    # bottom_right = (150, 150)  # 右下角坐标
    # # top_left_2 = (150, 150)  # 左上角坐标
    # # bottom_right_2 = (250, 250)  # 右下角坐标
    # # 将原始图像的相应区域替换为生成的噪声
    # image_temp = gray_batch.detach()
    # Temp_z = (1-z).detach()
    # image_temp[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = Temp_z[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]


    fz = model_Rec(z)
   
    f_z  = efficient_ti_model.get_image_embeddings(
            fz.detach().cpu()
        ).detach()
    f_z = f_z.to(device, memory_format=torch.contiguous_format)
    ff_z = model_Rec(f_z)
    return gray_batch, fz , ff_z

# 归一化函数
def normalize_with_clipping(image, lower_percentile=5, upper_percentile=95):
    # 计算上下分位数
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    
    # 剪切
    image_clipped = np.clip(image, lower_bound, upper_bound)
    
    # 归一化
    image_normalized = (image_clipped - image_clipped.min()) / (image_clipped.max() - image_clipped.min())
    return image_normalized

if __name__ == "__main__":

    dataloader = data_loader_my()

    image_temp , fz , ff_z = model_test(dataloader)
    # ff_z = model_Rec(ff_z)
    # 显示原始图像和损坏后的图像
    original_image = image_temp.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    corrupted_image_out = fz[0].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    gradient_image = ff_z[0].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

    # 归一化图像
    original_image = normalize_with_clipping(original_image)
    corrupted_image_out = normalize_with_clipping(corrupted_image_out)
    gradient_image = normalize_with_clipping(gradient_image)

    # 创建图形
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, vmin=0, vmax=1)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(corrupted_image_out, vmin=0, vmax=1)
    plt.title("Corrupted Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gradient_image, vmin=0, vmax=1)
    plt.title("Gradient Image")
    plt.axis('off')

    # 显示图像
    plt.tight_layout()
    plt.show()
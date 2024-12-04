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

import matplotlib.pyplot as plt
from matplotlib import rcParams
import torchvision.transforms.functional as TF

from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

from torch.utils.tensorboard import SummaryWriter

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
                        phase='test',
                        data_transform=train_data_transform,
                        num_task=5)
    dataloader = DataLoader(dataset, batch_size=3, num_workers=12,
                            sampler=SubsetRandomSampler(dataset.sample_indices_in_task[0]), drop_last=False)
    return dataloader

model_Rec = ReconstructiveSubNetworkOri(in_channels=3, out_channels=3)
model_Rec.to(device)
model_Rec.apply(weights_init)
model_Rec.load_state_dict(torch.load(os.path.join('./Ano_SAM_only_1', f"epoch_20.pckl"), map_location='cuda:0'))
efficient_ti_model = build_efficient_sam_vitt().requires_grad_(False)

def model_train(dataloader):
    num_epochs = 100 #训练参数
    lastepoch = 0 #上次训练参数
    #参数初始化
    optimizer = torch.optim.Adam([{"params": model_Rec.parameters(), "lr": 0.0001}, ], weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [(num_epochs+lastepoch) * 0.8, (num_epochs+lastepoch) * 0.9], gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    n_iter = 0
    # 创建 TensorBoard SummaryWriter
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    for e in range(num_epochs):
        model_Rec.train()
        train_loss = 0.0
        epoch = e + lastepoch  + 1
        print("Epoch: " + str(epoch))
        for i_batch, sample_batched in enumerate(dataloader):
            
            gray_batch = sample_batched["img"].clone()
            bsz = gray_batch.shape[0]

            z_ori  = efficient_ti_model.get_image_embeddings(
                gray_batch.detach()
                ).detach()
            z_ori = z_ori.to(device, memory_format=torch.contiguous_format)

            fx = model_Rec(z_ori)
            gray_batch_GPU = gray_batch.to(device, memory_format=torch.contiguous_format)
            ssim_loss = loss_ssim(fx, gray_batch_GPU)
            l2_loss = loss_l2(fx, gray_batch_GPU)
            
            loss = ssim_loss +l2_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bsz

            n_iter += 1
        train_loss /= len(sample_batched["img"])
        writer.add_scalar('Loss/train', train_loss, epoch)  # 记录损失
        print(f"Epoch {epoch} loss: {train_loss:.4f}")
        scheduler.step()
        if epoch % 10 == 0 or e == num_epochs - 1:
            torch.save(model_Rec.state_dict(), os.path.join('./Ano_SAM_only_1', f"epoch_{epoch}.pckl"))
    writer.close()
    print("Train_ok")

def model_test(dataloader):
    
    model_Rec.train()
    i = 20
    # for i_batch, sample_batched in enumerate(dataloader):
    sample_batched = dataloader.dataset[i]
    gray_batch = sample_batched["img"].clone()
    z  = efficient_ti_model.get_image_embeddings(
            gray_batch[None, ...]
        ).detach()
    z = z.to(device, memory_format=torch.contiguous_format)

    fz = model_Rec(z)
   
    return gray_batch, fz 

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
    # model_train(dataloader)
    image_temp , fz = model_test(dataloader)
    # ff_z = model_Rec(ff_z)
    # 显示原始图像和损坏后的图像
    original_image = image_temp.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    corrupted_image_out = fz[0].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    gradient_image = fz[0].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

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
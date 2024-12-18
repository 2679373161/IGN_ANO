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

def get_freq_means_and_stds(x: Tensor) -> Tuple[Tensor]:
    freq = torch.fft.fft2(x)
    real_mean = freq.real.mean(dim=0)
    real_std = freq.real.std(dim=0)
    imag_mean = freq.imag.mean(dim=0)
    imag_std = freq.imag.std(dim=0)
    return real_mean, real_std, imag_mean, imag_std

def get_noise(
    real_mean: Tensor,
    real_std: Tensor,
    imag_mean: Tensor,
    imag_std: Tensor,
) -> Tensor:
    freq_real = torch.normal(real_mean, real_std)
    freq_imag = torch.normal(imag_mean, imag_std)
    freq = freq_real + 1j * freq_imag
    noise = torch.fft.ifft2(freq)
    return noise.real

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
    dataloader = DataLoader(dataset, batch_size=2, num_workers=16,
                            sampler=SubsetRandomSampler(dataset.sample_indices_in_task[0]), drop_last=False)
    return dataloader

model_Rec = ReconstructiveSubNetworkOri(in_channels=3, out_channels=3)
model_Rec_copy = ReconstructiveSubNetworkOri(in_channels=3, out_channels=3).requires_grad_(False)
model_Rec.to(device)
model_Rec_copy.to(device)
model_Rec.apply(weights_init)
# model_Rec.load_state_dict(torch.load(os.path.join('./Ano_Try_1', f"epoch_200.pckl"), map_location='cuda:0'))
efficient_ti_model = build_efficient_sam_vitt().requires_grad_(False)

def model_train(dataloader):
    num_epochs = 20 #训练参数
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
            
            gray_batch = sample_batched["img"].clone()
            bsz = gray_batch.shape[0]

            if bsz <= 1:
                continue

            freq_means_and_stds = get_freq_means_and_stds(gray_batch)
            gray_batch_FF = torch.stack([get_noise(*freq_means_and_stds) for _ in range(bsz)])
            gray_batch_FF = gray_batch_FF.to(device, memory_format=torch.contiguous_format)

            z_ori  = efficient_ti_model.get_image_embeddings(
                gray_batch.detach()
                ).detach()
            z_ori = z_ori.to(device, memory_format=torch.contiguous_format)
            
            z_FF  = efficient_ti_model.get_image_embeddings(
                gray_batch_FF.cpu().detach()
                ).detach()
            z_FF = z_FF.to(device, memory_format=torch.contiguous_format)
    
            
            model_Rec_copy.load_state_dict(model_Rec.state_dict())
            fx = model_Rec(z_ori)
            fz = model_Rec(z_FF)
            f_z = fz.detach()
            f_z_FF = efficient_ti_model.get_image_embeddings(
                f_z.cpu().detach()
                ).detach().to(device, memory_format=torch.contiguous_format)
            ff_z = model_Rec(f_z_FF)
            fz_FF = efficient_ti_model.get_image_embeddings(
                fz.cpu().detach()
                ).detach().to(device, memory_format=torch.contiguous_format)
            f_fz = model_Rec_copy(fz_FF)
            # compute losses
            gray_batch = gray_batch.to(device, memory_format=torch.contiguous_format)
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
        if epoch % 1 == 0 or e == num_epochs - 1:
            torch.save(model_Rec.state_dict(), os.path.join('./Ano_SAM_1', f"epoch_{epoch}.pckl"))
    print("Train_ok")

def model_test(dataloader):
    kernel = torch.ones(3, 3).to(device)
    model_Rec.train()
    i = 0
    # for i_batch, sample_batched in enumerate(dataloader):
    sample_batched = dataloader.dataset[i]
    gray_batch = sample_batched["img"].clone()
    z  = efficient_ti_model.get_image_embeddings(
            gray_batch[None, ...]
        ).detach()
    z = z.to(device, memory_format=torch.contiguous_format)

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
    model_train(dataloader)
    # image_temp , fz , ff_z = model_test(dataloader)
    # # ff_z = model_Rec(ff_z)
    # # 显示原始图像和损坏后的图像
    # original_image = image_temp.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    # corrupted_image_out = fz[0].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    # gradient_image = ff_z[0].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

    # # 归一化图像
    # original_image = normalize_with_clipping(original_image)
    # corrupted_image_out = normalize_with_clipping(corrupted_image_out)
    # gradient_image = normalize_with_clipping(gradient_image)

    # # 创建图形
    # plt.figure(figsize=(15, 5))

    # plt.subplot(1, 3, 1)
    # plt.imshow(original_image, vmin=0, vmax=1)
    # plt.title("Original Image")
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.imshow(corrupted_image_out, vmin=0, vmax=1)
    # plt.title("Corrupted Image")
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.imshow(gradient_image, vmin=0, vmax=1)
    # plt.title("Gradient Image")
    # plt.axis('off')

    # # 显示图像
    # plt.tight_layout()
    # plt.show()
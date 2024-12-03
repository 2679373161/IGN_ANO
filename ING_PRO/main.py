from mvtec2d import MVTec2D
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from AEUnet import ReconstructiveSubNetwork
from torch import Tensor
import torch
from torch import optim
from loss import SSIM
import os
from PIL import Image
import numpy as np
from typing import Any, List, Optional, Tuple, Union
import torch.nn.functional as fn
import kornia

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x

def get_freq_means_and_stds(x: Tensor) -> Tuple[Tensor]:
    freq = torch.fft.fft2(x)
    real_mean = freq.real.mean(dim=0)
    real_std = freq.real.std(dim=0)
    imag_mean = freq.imag.mean(dim=0)
    imag_std = freq.imag.std(dim=0)
    return real_mean, real_std, imag_mean, imag_std

def get_freq_means_and_stds_T(x: Tensor,
                              real_std: Tensor,
                              imag_std: Tensor) -> Tuple[Tensor]:
    freq = torch.fft.fft2(x)
    real_mean = freq.real.mean(dim=0)
    # real_std_O = real_std.clone()
    imag_mean = freq.imag.mean(dim=0)
    # imag_std_0 = imag_std.clone()
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
def get_noise_constant(
    real_mean: Tensor,
    imag_mean: Tensor,
        gain: int
) -> Tensor:
    real_std = torch.ones(real_mean.shape)*gain
    imag_std = torch.ones(imag_mean.shape)*gain
    dtype = real_std.dtype
    real_std = torch.as_tensor(real_std, dtype=dtype, device=real_mean.device)
    imag_std = torch.as_tensor(imag_std, dtype=dtype, device=imag_mean.device)
    freq_real = torch.normal(real_mean, real_std)
    freq_imag = torch.normal(imag_mean, imag_std)
    freq = freq_real + 1j * freq_imag
    noise = torch.fft.ifft2(freq)
    return noise.real

def unnormalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    """Unnormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def Get_feature_map(InputImg: Tensor,kernel: Tensor) -> Tensor:
    Temp_img_new = unnormalize(InputImg, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    gradient_new = kornia.morphology.gradient(Temp_img_new, kernel)

    freq_means_and_stds = get_freq_means_and_stds(gradient_new)
    noise = get_noise_constant(freq_means_and_stds[0], freq_means_and_stds[2],50)

    return noise
def main():
    test_mode = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_transform = T.Compose([T.Resize((256, 256)),
                               T.CenterCrop(256),
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    dataloader = DataLoader(dataset, batch_size=4, num_workers=16,
                            sampler=SubsetRandomSampler(dataset.sample_indices_in_task[0]), drop_last=False)

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    # model_copy = ReconstructiveSubNetwork(in_channels=3, out_channels=3).requires_grad_(False)
    # model.load_state_dict(torch.load(os.path.join('.', "test.pckl"), map_location='cuda:0'))
    model.to(device)
    # model_copy.to(device)
    model.apply(weights_init)

    num_epochs = 360
    lastepoch = 40
    model.load_state_dict(torch.load(os.path.join('.', f"epoch_{lastepoch}.pckl"), map_location='cuda:0'))

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 0.0001}, ], weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [num_epochs * 0.8, num_epochs * 0.9], gamma=0.2, last_epoch=-1)
    rec_weight = 20
    idem_weight = 20
    tight_weight = 2.5
    idem_weight /= rec_weight
    tight_weight /= rec_weight
    loss_tight_clamp_ratio = 1.5
    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    # Feature_means_and_stds_Image = None
    # for i_batch, sample_batched in enumerate(dataloader):
    #     if i_batch >20:
    #         break
    #
    #     if i_batch == 0:
    #         Feature_means_and_stds_Image = sample_batched["img"].clone()
    #     else:
    #         Feature_means_and_stds_Image = torch.cat((sample_batched["img"], Feature_means_and_stds_Image), dim=0)
    #
    # Feature_means_and_stds = get_freq_means_and_stds(Feature_means_and_stds_Image)

    n_iter = 0
    kernel = torch.ones(3, 3).to(device)
    for e in range(num_epochs):
        model.train()
        train_loss = 0.0
        epoch = e + lastepoch  + 1
        print("Epoch: " + str(epoch))
        for i_batch, sample_batched in enumerate(dataloader):
            # print("i_batch: " + str(i_batch))

            gray_batch = sample_batched["img"].clone().to(device)
            bsz = gray_batch.shape[0]
            # freq_means_and_stds = get_freq_means_and_stds_T(gray_batch, (Feature_means_and_stds[1]).to(device), (Feature_means_and_stds[3]).to(device))
            # z = torch.stack([get_noise(*freq_means_and_stds) for _ in range(bsz)])
            z = torch.stack([Get_feature_map(gray_batch,kernel) for _ in range(bsz)])
            z = z.to(device, memory_format=torch.contiguous_format)

            # compute model outputs
            # model_copy.load_state_dict(model.state_dict())
            # fx = model(gray_batch)
            # fz = model(z)
            # f_z = fz.detach()
            # ff_z = model(f_z)
            # f_fz = model_copy(fz)
            #
            # loss_rec = fn.l1_loss(fx, gray_batch, reduction="none").view(bsz, -1).mean(dim=-1)
            # loss_idem = fn.l1_loss(f_fz, fz, reduction="mean")
            # loss_tight = -fn.l1_loss(ff_z, f_z, reduction="none").view(bsz, -1).mean(dim=-1)
            # loss_tight_clamp = loss_tight_clamp_ratio * loss_rec
            # loss_tight = fn.tanh(loss_tight / loss_tight_clamp) * loss_tight_clamp
            # loss_rec = loss_rec.mean()
            # loss_tight = loss_tight.mean()
            #
            # loss = loss_rec + idem_weight * loss_idem + tight_weight * loss_tight
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            gray_rec = model(z)
            l2_loss = loss_l2(gray_rec, gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)
            loss = l2_loss + ssim_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            # n_iter += 1
            train_loss += loss.item() * bsz
            n_iter += 1
            # with torch.no_grad():
            #     # image_T = kornia.tensor_to_image((1 - z).squeeze(0))  # Tensor to image
            #     recimg = (1 - z).squeeze(0).cpu().float().numpy()
            #     recimg = (recimg.transpose(1, 2, 0) * 255.).astype(np.uint8)
            #     Img_test_11 = Image.fromarray(recimg)
            #     Img_test_11.save(f'Z_Ori_IN_{i_batch}.png')

        train_loss /= len(sample_batched["img"])
        print(f"Epoch {epoch} loss: {train_loss:.4f}")

        scheduler.step()
        if epoch % 20 == 0 or e == num_epochs - 1:
            torch.save(model.state_dict(), os.path.join('.', f"epoch_{epoch}.pckl"))


    if test_mode:
        model.load_state_dict(torch.load(os.path.join('.', "epoch_40.pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()
        dataset_test  = MVTec2D(data_path='D:/Myself/Postgraduate/07_Paper/Python_Prj/MLDFR/MVTec',
                          learning_mode='centralized',
                          phase='test',
                          data_transform=train_data_transform,
                          num_task=1)
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=16,
                                sampler=SubsetRandomSampler(dataset_test.sample_indices_in_task[0]), drop_last=False)

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader_test):
                if i_batch < 8:
                    gray_batch = sample_batched["img"].clone().to(device)
                    bsz = gray_batch.shape[0]
                    # freq_means_and_stds = get_freq_means_and_stds_T(gray_batch, (Feature_means_and_stds[1]).to(device),
                    #                                                 (Feature_means_and_stds[3]).to(device))
                    # z = torch.stack([get_noise(*freq_means_and_stds) for _ in range(bsz)])
                    z = torch.stack([Get_feature_map(gray_batch,kernel) for _ in range(bsz)])
                    z = z.to(device, memory_format=torch.contiguous_format)
                    gray_rec = model(z)
                    # z_out_1= model(z)
                    # z_out_2 = model(z_out_1)
                    #     # denormalize
                    #     images = [img / 2 + 0.5 for img in images]
                    #     # to numpy arrays
                    #     images = [img.cpu().permute(1, 2, 0).float().numpy() for img in images]
                    #     # to PIL image
                    #     images = [(255 * img).round().astype("uint8") for img in images]
                    #     images = [Image.fromarray(img) for img in images]
                    recimg = gray_rec.squeeze(0).cpu().float().numpy()
                    recimg = denormalization(recimg)
                    oriimg = gray_batch.squeeze(0).cpu().float().numpy()
                    oriimg = denormalization(oriimg)
                    noiseimg = (1 - z).squeeze(0).cpu().float().numpy()
                    noiseimg = (noiseimg.transpose(1, 2, 0) * 255.).astype(np.uint8)
                    # z_out_img_1 = z_out_1.squeeze(0).cpu().float().numpy()
                    # z_out_img_1 = (z_out_img_1.transpose(1, 2, 0) * 255.).astype(np.uint8)
                    # z_out_img_2 = z_out_2.squeeze(0).cpu().float().numpy()
                    # z_out_img_2 = (z_out_img_2.transpose(1, 2, 0) * 255.).astype(np.uint8)

                    img_Test_T_O = recimg.copy()
                    Img_test_1223 = Image.fromarray(img_Test_T_O)
                    Img_test_1223.save(f'Output_Ori_{i_batch}.png')

                    img_Test_T_2 = oriimg.copy()
                    Img_test_1224 = Image.fromarray(img_Test_T_2)
                    Img_test_1224.save(f'Input_{i_batch}.png')

                    img_Test_T_3 = noiseimg.copy()
                    Img_test_1225 = Image.fromarray(img_Test_T_3)
                    Img_test_1225.save(f'noiseimg_{i_batch}.png')

                    # img_Test_T_4 = z_out_img_1.copy()
                    # Img_test_1226 = Image.fromarray(img_Test_T_4)
                    # Img_test_1226.save(f'z_out_img_1_{i_batch}.png')
                    #
                    # img_Test_T_5 = z_out_img_2.copy()
                    # Img_test_1227 = Image.fromarray(img_Test_T_5)
                    # Img_test_1227.save(f'z_out_img_2_{i_batch}.png')
                else:
                    break
            print("all_ok")
    else:
        print("Train_ok")


if __name__ == '__main__':
    main()
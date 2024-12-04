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

from msgms import MSGMSLoss
from ColorlossLab import ColorDifference
from utils import savefig
from sklearn.metrics import roc_auc_score, average_precision_score
from generic_util import trapezoid
from pro_curve_util import compute_pro

def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return fn.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def normalize_with_clipping_Uint8(image, lower_percentile=5, upper_percentile=95):
    # 计算上下分位数
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    
    # 剪切
    image_clipped = np.clip(image, lower_bound, upper_bound)
    
    # 归一化
    image_normalized = ((image_clipped - image_clipped.min()) / (image_clipped.max() - image_clipped.min())* 255.).astype(np.uint8)
    return image_normalized

def test(obj_names, mvtec_path, checkpoint_path, base_model_name,saveimages):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_transform = T.Compose([T.Resize((256, 256)),
                T.CenterCrop(256),
                T.ToTensor(),
                ])

    mask_transform = T.Compose([T.Resize(256),
                T.CenterCrop(256),
                T.ToTensor(),
                ])
    train_data_transform = img_transform, mask_transform

    img_dim =256
    modelOri = ReconstructiveSubNetworkOri(in_channels=3, out_channels=3)
    modelOri.to(device)
    modelOri.load_state_dict(torch.load(os.path.join('./Ano_SAM_1/', "epoch_20.pckl"), map_location='cuda:0'))
    modelOri.cuda()
    # modelOri.eval()
    efficient_ti_model = build_efficient_sam_vitt().requires_grad_(False)

    dataset_test  = MVTec2D(data_path='D:/Myself/Postgraduate/07_Paper/Python_Prj/MLDFR/MVTec',
                        learning_mode='centralized',
                        phase='test',
                        data_transform=train_data_transform,
                        num_task=1)
    
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=16,
                            sampler=SubsetRandomSampler(dataset_test.sample_indices_in_task[0]), drop_last=False)
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_pro_list = []
    obj_auroc_image_list = []
    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataloader_test)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataloader_test)))
    mask_cnt = 0

    # calculate pro
    pro_gt = []
    pro_out = []
    anomaly_score_gt = []
    anomaly_score_prediction = []
    msgms = MSGMSLoss().to(device)
    with torch.no_grad():
        i = 0
        for i_batch, sample_batched in enumerate(dataloader_test):

            gray_batch = sample_batched["img"].clone()
            z  = efficient_ti_model.get_image_embeddings(
                    gray_batch
                ).detach()
            z = z.to(device, memory_format=torch.contiguous_format)

            fz = modelOri(z)

            f_z  = efficient_ti_model.get_image_embeddings(
                    fz.detach().cpu()
                ).detach()
            f_z = f_z.to(device, memory_format=torch.contiguous_format)
            z_out_1 = modelOri(f_z)
            

            is_normal = sample_batched["label"].detach().numpy()
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))


            recimg = z_out_1.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
            recimg = normalize_with_clipping_Uint8(recimg)
            oriimg = gray_batch.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
            oriimg = normalize_with_clipping_Uint8(oriimg)

            colorD = ColorDifference(recimg, oriimg)

            mgsgmsmap = msgms(z_out_1.to(device, memory_format=torch.contiguous_format), 
                              gray_batch.to(device, memory_format=torch.contiguous_format), as_loss=False)
            mgsgmsmapmean = mean_smoothing(mgsgmsmap, 21)
            out_mask_gradient = mgsgmsmapmean.detach().cpu().numpy()

            # combined
            out_mask_averaged = colorD[None, None, :, :] + out_mask_gradient

            # '''save result images
            if saveimages:
                segresult = out_mask_averaged[0, 0, :, :]
                truemaskresult = true_mask[0, 0, :, :]
 
                # savefig(gray_batch, segresult, truemaskresult, f'{"."}/{"IGN_LessTop_Mode"}/' + f'segresult{i}.png',
                #         fz)
                savefig(gray_batch, segresult, truemaskresult, f'{"."}/{"IGN_SAM_Mode"}/' + f'segresult_2R{i}.png',
                        z_out_1)
                i = i + 1
            # '''

            image_score = np.max(out_mask_averaged)
            anomaly_score_prediction.append(image_score)
            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_averaged.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[
            mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

            # for pro
            truegt = true_mask_cv[:, :, 0]
            outresult = out_mask_averaged[0, 0, :, :]
            pro_gt.append(truegt)
            pro_out.append(outresult)



        all_fprs, all_pros = compute_pro(
            anomaly_maps=pro_out,
            ground_truth_maps=pro_gt,
            num_thresholds=5000)

        au_pro = trapezoid(all_fprs, all_pros, x_max=0.3)
        au_pro /= 0.3
        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        obj_pro_list.append(au_pro)
        print("bottle")
        print("AUC Image:  " + str(auroc))
        print("AP Image:  " + str(ap))
        print("AUC Pixel:  " + str(auroc_pixel))
        print("AP Pixel:  " + str(ap_pixel))
        print("PRO:  " + str(au_pro))

        print("==============================")
        print("all_ok")

if __name__ == "__main__":
    test( '','','','',1)
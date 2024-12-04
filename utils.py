import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


def normalize_with_clipping(image, lower_percentile=5, upper_percentile=95):
    # 计算上下分位数
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    
    # 剪切
    image_clipped = np.clip(image, lower_bound, upper_bound)
    
    # 归一化
    image_normalized = (image_clipped - image_clipped.min()) / (image_clipped.max() - image_clipped.min())
    return image_normalized

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x
def savefig(img,outmask,truemask,path,reconimg):
    grid = ImageGrid(
        fig=plt.figure(figsize=(16, 4)),
        rect=111,
        nrows_ncols=(1, 4),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )
    img = img.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = normalize_with_clipping(img)

    reconimg = reconimg.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    reconimg = normalize_with_clipping(reconimg)

    grid[0].imshow(img)
    grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[0].set_title("Input Image", fontsize=14)

    grid[1].imshow(reconimg)
    grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[1].set_title("Recon Image", fontsize=14)

    grid[2].imshow(truemask)
    grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[2].set_title("GroundTruth", fontsize=14)

    grid[3].imshow(img)
    im = grid[3].imshow(outmask, alpha=0.3, cmap="jet")
    grid[3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[3].cax.colorbar(im)
    grid[3].cax.toggle_label(True)
    grid[3].set_title("Anomaly Map", fontsize=14)

    plt.savefig(path, bbox_inches="tight")
    plt.close()



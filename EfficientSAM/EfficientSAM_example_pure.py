from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
# from squeeze_sam.build_squeeze_sam import build_squeeze_sam

from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile
import matplotlib.pyplot as plt

def visualize_image_embeddings(embeddings, batched_images):
        """
        Visualizes the image embeddings obtained from the model.

        Arguments:
        model: The model instance that has the get_image_embeddings method.
        batched_images: A tensor of shape [B, 3, H, W].
        """
        # 创建图形以显示嵌入
        num_embeddings = len(embeddings)
        if num_embeddings == 1:  # 只有一个嵌入
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        else:  # 有多个嵌入
            fig, axes = plt.subplots(1, num_embeddings, figsize=(15, 5))

        # 处理每个嵌入
        for i, embedding in enumerate(embeddings):
            # 选择第一个样本
            embedding_sample = embedding  # 形状为 [256, 64, 64]

            # 显示前几个通道的嵌入
            num_channels_to_show = min(5, embedding_sample.shape[0])  # 最多显示 5 个通道

            for j in range(num_channels_to_show):
                channel_image = embedding_sample[j]  # 选择第 j 个通道
                
                # 检查通道的形状
                if channel_image.ndim != 2:
                    raise ValueError(f"Unexpected shape for channel {j + 1}: {channel_image.shape}. Expected 2D.")

                # 归一化到 [0, 1]
                channel_image = (channel_image - channel_image.min()) / (channel_image.max() - channel_image.min() + 1e-8)

                plt.subplot(num_embeddings, num_channels_to_show, i * num_channels_to_show + j + 1)
                plt.imshow(channel_image.detach().cpu().numpy(), cmap='gray')
                plt.title(f'Embedding {i + 1} - Channel {j + 1}')
                plt.axis('off')

        plt.tight_layout()
        plt.show()

# models = {}

# # Build the EfficientSAM-Ti model.
# models['efficientsam_ti'] = build_efficient_sam_vitt()

# # Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
# with zipfile.ZipFile("EfficientSAM/weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
#     zip_ref.extractall("EfficientSAM/weights")
# Build the EfficientSAM-S model.
# models['efficientsam_s'] = build_efficient_sam_vits()

# Build the SqueezeSAM model.
# models['squeeze_sam'] = build_squeeze_sam()



if __name__ == "__main__":
    # load an image
    sample_image_np = np.array(Image.open("figs/examples/dogs.jpg"))
    sample_image_tensor = transforms.ToTensor()(sample_image_np)
    # Feed a few (x,y) points in the mask as input.

    # input_points = torch.tensor([[[[580, 350], [650, 350]]]])
    # input_labels = torch.tensor([[[1, 1]]])

    efficient_ti_model = build_efficient_sam_vitt()
    embeddings  = efficient_ti_model.get_image_embeddings(
            sample_image_tensor[None, ...]
        ).detach()
    visualize_image_embeddings(embeddings, sample_image_tensor)

# Run inference for both EfficientSAM-Ti and EfficientSAM-S models.
# for model_name, model in models.items():
#     print('Running inference using ', model_name)
#     predicted_logits, predicted_iou = model(
#         sample_image_tensor[None, ...],
#         input_points,
#         input_labels,
#     )
#     sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
#     predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
#     predicted_logits = torch.take_along_dim(
#         predicted_logits, sorted_ids[..., None, None], dim=2
#     )
#     # The masks are already sorted by their predicted IOUs.
#     # The first dimension is the batch size (we have a single image. so it is 1).
#     # The second dimension is the number of masks we want to generate (in this case, it is only 1)
#     # The third dimension is the number of candidate masks output by the model.
#     # For this demo we use the first mask.
#     mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
#     masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
#     Image.fromarray(masked_image_np).save(f"figs/examples/dogs_{model_name}_mask.png")
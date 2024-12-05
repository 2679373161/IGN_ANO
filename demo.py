from flask import Flask, request, render_template, send_file
import torch
from PIL import Image
from torchvision import transforms as T

from ING_PRO.AEUnet_SAM import ReconstructiveSubNetwork as ReconstructiveSubNetworkOri
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt
import io
import numpy as np
import os

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelOri = ReconstructiveSubNetworkOri(in_channels=3, out_channels=3)
modelOri.to(device)

modelOri.cuda()
# modelOri.eval()
efficient_ti_model = build_efficient_sam_vitt().requires_grad_(False).to(device)

def load_model(model_name):
    if model_name == "Grid":
        modelOri.load_state_dict(torch.load(os.path.join('./Ano_SAM_1/grid/', "epoch_50.pckl"), map_location='cuda:0'))
    elif model_name == "Bottle":
        modelOri.load_state_dict(torch.load(os.path.join('./Ano_SAM_1/bottle/', "epoch_20.pckl"), map_location='cuda:0'))
    elif model_name == "ALL":
        modelOri.load_state_dict(torch.load(os.path.join('./Ano_SAM_1/ALL/', "epoch_5.pckl"), map_location='cuda:0'))


def normalize_with_clipping(image, lower_percentile=5, upper_percentile=95):
    # 计算上下分位数
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    
    # 剪切
    image_clipped = np.clip(image, lower_bound, upper_bound)
    
    # 归一化
    image_normalized = (image_clipped - image_clipped.min()) / (image_clipped.max() - image_clipped.min())
    return (image_normalized* 255).astype(np.uint8)  # Convert to uint8

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    model_param = request.form['model_params']  # Get the selected model parameters

    load_model(model_param)  # Load the selected model

    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        img = Image.open(file.stream).convert('RGB')

        # Preprocess the image
        # image = preprocess(Image.open(image_path)).unsqueeze(0)
        img_transform = T.Compose([T.Resize((256, 256)),
                    T.CenterCrop(256),
                    T.ToTensor(),
                    ])
        img_tensor = img_transform(img).unsqueeze(0)
        
        gray_batch = img_tensor.to(device, memory_format=torch.contiguous_format)
        z  = efficient_ti_model.get_image_embeddings(
                    gray_batch
                ).detach()

        fz = modelOri(z)
   

        img_out = fz[0].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        img_out = normalize_with_clipping(img_out)

        # Convert output image to BytesIO
        output_image = Image.fromarray(img_out)
        img_bytes = io.BytesIO()
        output_image.save(img_bytes, format='PNG')  # Save as PNG
        img_bytes.seek(0)  # Rewind the BytesIO object for reading

    return send_file(img_bytes, mimetype='image/png')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    app.run(debug=True)
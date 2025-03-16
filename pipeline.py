import sys
sys.path.append('..')
from pathlib import Path
from raw_prc_pipeline.pipeline import RawProcessingPipelineDemo
import cv2
from raw_utils import fraction_from_json, json_read
from raw_prc_pipeline import expected_landscape_img_height, expected_landscape_img_width
import torch
import os
from SCUNet import SCUNet as net
import argparse
import numpy as np
import time
from IAT import IAT
import torchvision
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def lowlight(data_lowlight):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = IAT().to(device)
    model.load_state_dict(torch.load('./model_zoo/best_Epoch16_1024.pth'))
    model.eval()
    with torch.no_grad():
        start = time.time()
        _, _, enhanced_image = model(data_lowlight)
    end_time = (time.time() - start)
    print(end_time)

    return enhanced_image

def denoise(img):
    n_channels = 3

    model_path = './model_zoo/SCU_best_Epoch4.pth'

    model = net(in_nc=n_channels, config=[4, 4, 4, 4, 4, 4, 4], dim=64)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    with torch.no_grad():
        denoised_img = model(img)

    return denoised_img
def main():
    raw_path = './data/test/raw/'
    save_path_SCU = 'data/test/final_rendering/'
    file_names = os.listdir(raw_path)

    for file_name in file_names:
        if file_name.lower().endswith('.png'):
            file_path = os.path.join(raw_path, file_name)
            png_path = Path(file_path)

            # parse raw img
            raw_image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            # parse metadata
            metadata = json_read(png_path.with_suffix('.json'), object_hook=fraction_from_json)

            pipeline_params = {
                'tone_mapping': 'Flash', # options: Flash, Storm, Base, Linear, Drago, Mantiuk, Reinhard
                'illumination_estimation': 'gw', # ie algorithm, options: "gw", "wp", "sog", "iwp"
                'denoise_flg': True,
                'out_landscape_width': expected_landscape_img_width,
                'out_landscape_height': expected_landscape_img_height,
                "color_matrix": [1.06835938, -0.29882812, -0.14257812,
                                 -0.43164062,  1.35546875,  0.05078125,
                                 -0.1015625,   0.24414062,  0.5859375]
            }

            pipeline_demo = RawProcessingPipelineDemo(**pipeline_params)

            linearized_image = pipeline_demo.linearize_raw(raw_image, metadata)

            normalized_image = pipeline_demo.normalize(linearized_image, metadata)

            demosaic_image = pipeline_demo.demosaic(normalized_image, metadata)

            rot_image = pipeline_demo.fix_orientation(demosaic_image, metadata)

            flip_img = pipeline_demo.horizontal_flip(rot_image, metadata)

            undistorted_img = pipeline_demo.projective(flip_img, metadata)

            crop_img = pipeline_demo.crop_resize(undistorted_img, metadata)

            denoised_image = pipeline_demo.denoise(crop_img, metadata)

            white_balanced_image = pipeline_demo.white_balance(denoised_image, metadata)

            xyz_image = pipeline_demo.xyz_transform(white_balanced_image, metadata)

            srgb_image = pipeline_demo.srgb_transform(xyz_image, metadata)
            img_uint8 = pipeline_demo.to_uint8(srgb_image, metadata)
            srgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
            srgb = (srgb / 255.0).astype(np.float32)
            srgb = torch.tensor(srgb).permute(2, 0, 1).unsqueeze(0).to(device)

            # Apply lowlight enhancement
            img_L = lowlight(srgb)

            # Clamp and denoise without saving
            img_L_clamped = img_L.clamp(0, 1)
            img_D = denoise(img_L_clamped)

            file_name = file_name.replace('.png', '.jpg')
            torchvision.utils.save_image(img_D, os.path.join(save_path_SCU, file_name))

            print(f"{file_name} process done!")
if __name__ == '__main__':
    main()
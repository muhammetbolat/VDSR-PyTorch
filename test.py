# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""File description: Realize the verification function after model training."""
import os

import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import image_quality_assessment
from image_quality_assessment import PSNR, SSIM
import imgproc
from model import VDSR
from train import AverageMeter


def main() -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Initialize the super-resolution model
    model = VDSR().to(config.device)
    print("Build VDSR model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load VDSR model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    sr_results_dir = os.path.join("results", "test", config.exp_name, "sr")
    if not os.path.exists(sr_results_dir):
        os.makedirs(sr_results_dir)

    lr_results_dir = os.path.join("results", "test", config.exp_name, "lr")
    if not os.path.exists(lr_results_dir):
        os.makedirs(lr_results_dir)


    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the image evaluation index.
    psnr_model = PSNR(config.upscale_factor, False)
    ssim_model = SSIM(config.upscale_factor, False)
    psnrMeter = AverageMeter("PSNR", ":4.2f")
    ssimMeter = AverageMeter("SSIM", ":4.2f")

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.hr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])
        lr_image_path = os.path.join(config.lr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(hr_image_path)}`...")
        # Make high-resolution image
        hr_image = cv2.imread(hr_image_path)
        hr_image_height, hr_image_width = hr_image.shape[:2]
        hr_image_height_remainder = hr_image_height % 12
        hr_image_width_remainder = hr_image_width % 12
        hr_image = hr_image[:hr_image_height - hr_image_height_remainder, :hr_image_width - hr_image_width_remainder, ...]


        ################################################################################################################
        # dct operation
        # Use high-resolution image to make low-resolution image
        lr_image = imgproc.dropHighFrequencies(hr_image, 1 / config.upscale_factor)

        hr_image = hr_image.astype(np.float32) / 255.
        lr_image = lr_image.astype(np.float32) / 255.
        """
        ################################################################################################################
        # bicubic operation
        # Read a batch of image data

        # Use high-resolution image to make low-resolution image
        hr_image = hr_image.astype(np.float32) / 255.
        lr_image = imgproc.imresize(hr_image, 1 / config.upscale_factor)
        lr_image = imgproc.imresize(lr_image, config.upscale_factor)
        """
        ################################################################################################################

        # Convert BGR image to YCbCr image
        lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=False)
        hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=False)

        # Split YCbCr image data
        lr_y_image, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
        hr_y_image, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert Y image data convert to Y tensor data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

        # Cal PSNR & SSIM
        #total_psnr += 10. * torch.log10(1. / torch.mean((sr_y_tensor - hr_y_tensor) ** 2))
        #total_ssim = image_quality_assessment.ssim(sr_y_image, hr_y_image, 0, False)

        psnr = psnr_model.forward(sr_y_tensor, hr_y_tensor)
        psnrMeter.update(psnr.item())

        ssim = ssim_model.forward(sr_y_tensor, hr_y_tensor)
        ssimMeter.update(ssim.item())

        # Save image
        sr_y_image = imgproc.tensor2image(sr_y_tensor, range_norm=False, half=True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        lr_ycbcr_image = cv2.merge([lr_y_image, hr_cb_image, hr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        lr_image = imgproc.ycbcr2bgr(lr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)
        cv2.imwrite(lr_image_path, lr_image * 255.0)

    print(f"PSNR: {psnrMeter.avg:4.2f} dB")
    print(f"SSIM: {ssimMeter.avg:4.2f}")


if __name__ == "__main__":
    main()
